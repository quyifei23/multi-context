"""Owned channel GPU VA -> existing external Memory mapping -> existing CPU VMA.

No numeric CPU/GPU address equality shortcut. This deliberately supports only
the observed client-created context-share and shared USERD/ring Memory object.
Other layouts fail closed; nothing here requests a map or reads memory.
"""
from analyze_userd_mapping import require, birth, ok, overlaps, fd_signature


def resolve_ring(events, identity, member, userd, vmas):
    client, uuid = identity['hClient'], identity['gpu_uuid_hex']
    ca = birth(events, client, member['channel']['handle'])
    require(ca['vaspace'] == 0 and ca.get('context_share'), 'explicit_owned_context_share_required')
    cs = birth(events, client, ca['context_share'])
    require(cs['class'] == 0x9067 and cs['parent'] == identity['group']['handle'] and
            cs['seq'] < ca['seq'] and cs.get('ctxshare_vaspace'), 'ctxshare_vaspace_binding_missing')
    va = birth(events, client, cs['ctxshare_vaspace'])
    require(va['class'] == 0x90f1 and va['parent'] == identity['device']['handle'] and va['seq'] < cs['seq'],
            'vaspace_lifecycle_or_owner_mismatch')
    base, size = ca['gpfifo_gpu_va'], ca['gpfifo_entries'] * 8
    require(0 < ca['gpfifo_entries'] <= 4096 and base % 8 == 0 and
            ca['gpfifo_entries'] & (ca['gpfifo_entries'] - 1) == 0, 'unsupported_ring_layout')
    reg = [e for e in events if e['kind'] == 'uvm_register_vaspace' and ok(e) and
           e['client'] == client and e['vaspace'] == va['object'] and e['gpu_uuid_hex'] == uuid and e['seq'] > va['seq']]
    require(len(reg) == 1, 'uvm_vaspace_registration_missing_or_ambiguous')
    reg = reg[0]
    init = [e for e in events if e['kind'] == 'uvm_initialize' and e['fd'] == reg['fd'] and ok(e) and e['seq'] < reg['seq']]
    require(len(init) == 1 and init[0]['flags'] == 0, 'uvm_owned_single_process_fd_required')
    init = init[0]
    mm = [e for e in events if e['kind'] == 'uvm_mm_initialize' and e['uvm_fd'] == reg['fd'] and
          e.get('rc', 0) == 0 and e['seq'] > init['seq']]
    # Pinned uvm.c:78-83 explicitly returns NV_WARN_NOTHING_TO_DO when
    # va_space_mm is disabled: the secondary FD may then be released. This is
    # not an ignored error or weakened owner/UUID gate. Other statuses fail.
    require(len(mm) == 1 and mm[0]['status'] in (0, 0x10006), 'uvm_mm_lifetime_binding_missing')
    live_fds = (reg['fd'], mm[0]['fd']) if mm[0]['status'] == 0 else (reg['fd'],)
    require(not any(e['kind'] == 'close' and ok(e) and e['seq'] > init['seq'] and
                    e['fd'] in live_fds for e in events), 'uvm_fd_closed_or_reused')
    channel_regs = [e for e in events if e['kind'] == 'uvm_register_channel' and ok(e) and
                    e['client'] == client and e['channel'] == ca['object'] and e['gpu_uuid_hex'] == uuid and
                    e['fd'] == reg['fd'] and e['seq'] > ca['seq']]
    require(len(channel_regs) == 1, 'channel_not_bound_to_uvm_vaspace')
    cr = channel_regs[0]
    for e in (reg, cr):
        require(fd_signature(e) == fd_signature(init) and e['rm_fd'] == ca['fd'] and
                fd_signature(e, 'rm_') == fd_signature(ca), 'uvm_rm_fd_binding_mismatch')
    require(not any(e['seq'] > reg['seq'] and e.get('fd') == reg['fd'] and
                    e['kind'] in ('uvm_register_vaspace', 'uvm_unregister_vaspace') and
                    e.get('gpu_uuid_hex') == uuid and ok(e) for e in events), 'uvm_vaspace_changed')
    require(not any(e['seq'] > cr['seq'] and e['kind'] == 'uvm_unregister_channel' and ok(e) and
                    e['client'] == client and e['channel'] == ca['object'] for e in events), 'uvm_channel_unregistered')
    maps = [e for e in events if e['kind'] == 'uvm_map_external' and ok(e) and e['fd'] == reg['fd'] and
            e['client'] == client and e['base'] <= base and base + size <= e['base'] + e['length'] and
            any(a['gpu_uuid_hex'] == uuid for a in e['gpu_attributes'])]
    require(len(maps) == 1, 'gpu_va_mapping_missing_or_ambiguous')
    gm = maps[0]
    require(gm['seq'] > reg['seq'] and gm['rm_fd'] == ca['fd'] and
            fd_signature(gm, 'rm_') == fd_signature(ca) and fd_signature(gm) == fd_signature(init), 'gpu_va_map_fd_mismatch')
    require(len(gm['gpu_attributes']) == 1 and gm['gpu_attributes'][0]['gpu_uuid_hex'] == uuid and
            gm['gpu_attributes'][0]['mapping_type'] in (1, 2, 3), 'gpu_va_map_uuid_or_access')
    # Reuse the already fully validated Memory generation and CPU mapping. No
    # attempt to map another object, guess a UVA, or scan unrelated host pages.
    require(gm['memory'] == ca['userd_handles'][0], 'ring_not_in_validated_userd_memory_object')
    ma = birth(events, client, gm['memory'])
    require(ma['seq'] == userd['memory_alloc_seq'] and ma['seq'] < gm['seq'], 'ring_memory_generation_changed')
    ranges = [e for e in events if e['kind'] == 'uvm_create_range' and ok(e) and e['fd'] == reg['fd'] and
              init['seq'] < e['seq'] < gm['seq'] and e['base'] == gm['base'] and e['length'] == gm['length']]
    require(len(ranges) == 1, 'external_range_lifetime_missing_or_ambiguous')
    for e in events:
        if e['seq'] <= gm['seq'] or not ok(e) or e.get('fd') != reg['fd']:
            continue
        if e['kind'] == 'uvm_ioctl':
            raise ValueError('unreviewed_uvm_operation_after_mapping')
        if e['kind'] == 'uvm_free' and e['base'] == ranges[0]['base']:
            raise ValueError('uvm_range_freed')
        if (e['kind'] in ('uvm_unmap_external', 'uvm_other_range', 'uvm_map_external', 'uvm_create_range') and
                overlaps(base, size, e['base'], e['length'])):
            raise ValueError('gpu_va_range_changed_after_mapping')
    setup = next(e for e in events if e['seq'] == userd['rm_map_seq'])
    cpu_map = next(e for e in events if e['seq'] == userd['mmap_seq'])
    offset = gm['offset'] + base - gm['base']
    require(setup['offset'] <= offset and offset + size <= setup['offset'] + setup['length'], 'ring_outside_existing_cpu_mapping')
    cpu = cpu_map['address'] + offset - setup['offset']
    require(cpu == userd['userd_cpu_address'] + offset - userd['userd_offset'] and cpu % 8 == 0, 'ring_memory_offset_mismatch')
    live = [v for v in vmas if v['start'] <= cpu and cpu + size <= v['end'] and v['perms'].startswith('r') and
            v['perms'].endswith('s') and v['inode'] == cpu_map['fd_ino'] and v['path'] == cpu_map['fd_path'] and
            v['file_offset'] + cpu - v['start'] == cpu - cpu_map['address']]
    require(len(live) == 1, 'ring_existing_readable_vma_missing')
    return dict(ring_cpu_address=cpu, ring_entries=ca['gpfifo_entries'], ring_memory_offset=offset,
                memory_generation=userd['memory_generation'], ctxshare_alloc_seq=cs['seq'], vaspace_alloc_seq=va['seq'],
                gpu_map_seq=gm['seq'], uvm_register_seq=reg['seq'], uvm_channel_seq=cr['seq'],
                uvm_mm_initialize_status=mm[0]['status'],
                memory_alloc_seq=ma['seq'], rm_map_seq=setup['seq'], mmap_seq=cpu_map['seq'])

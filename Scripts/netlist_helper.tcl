## Function to check if a cluster is valid or not
## A cluster is valid clip if the clip utilization 
## i.e. the smallest bounding box of the cluster is
## more than the threshold. 
proc is_valid_cluster { cluster_id {threshold 0.4}} {
  set cluster_ptr [dbget top.fplan.groups.name ${cluster_id} -p]
  ## Get cluster bbox
  set llx [lindex [lsort -real [dbget ${cluster_ptr}.members.box_llx ]] 0]
  set lly [lindex [lsort -real [dbget ${cluster_ptr}.members.box_lly ]] 0]
  set urx [lindex [lsort -real [dbget ${cluster_ptr}.members.box_urx ]] end]
  set ury [lindex [lsort -real [dbget ${cluster_ptr}.members.box_ury ]] end]
  set inst_count [llength [dbget ${cluster_ptr}.members]]
  set area [expr {($urx-$llx)*($ury-$lly)}]
  
  ## Get inst area
  set inst_area [expr [join [dbget ${cluster_ptr}.members.area] + ] ]
  set clip_util [expr {$inst_area/$area}]
  
  ## If clip utilization is greater than threshold then 
  ## return bbox1 else return bbox0
  if {$clip_util < $threshold} {
    set bbox [list 0 $llx $lly $urx $ury $clip_util $inst_count]
  } else {
    set bbox [list 1 $llx $lly $urx $ury $clip_util $inst_count]
  }
  return $bbox
}



proc update_placement_at_cluster_center { cluster_ptr } {
  set cluster_id [dbget ${cluster_ptr}.name]
  set box [is_valid_cluster $cluster_id]
  set llx [lindex $box 1]
  set lly [lindex $box 2]
  set urx [lindex $box 3]
  set ury [lindex $box 4]
  set center_x [expr {($urx+$llx)/2.0}]
  set center_y [expr {($ury+$lly)/2.0}]

  ## Place each instance at the center of the cluster
  foreach inst_ptr [dbget ${cluster_ptr}.members] {
    set inst_width [dbget ${inst_ptr}.box_sizex]
    set inst_height [dbget ${inst_ptr}.box_sizey]
    set inst_x [expr {$center_x - $inst_width/2.0}]
    set inst_y [expr {$center_y - $inst_height/2.0}]
    dbset ${inst_ptr}.pt_x $inst_x
    dbset ${inst_ptr}.pt_y $inst_y
  }
}

proc gen_seeded_init_place {} {
  foreach group_ptr [dbget top.fplan.groups] {
    update_placement_at_cluster_center $group_ptr
  }
}

## Procedure that writes out instance details for a given cluster
proc write_inst_header { fp } {
  puts -nonewline $fp "Inst_name,Master_name,Inst_width,Inst_height,Inst_x"
  puts $fp ",Inst_y,Inst_orient"
}

proc write_inst_details { fp inst_ptr } {
  set inst_name [lindex [dbget ${inst_ptr}.name] 0]
  set inst_master [dbget ${inst_ptr}.cell.name]
  set inst_width [dbget ${inst_ptr}.box_sizex]
  set inst_height [dbget ${inst_ptr}.box_sizey]
  set inst_x [dbget ${inst_ptr}.box_llx]
  set inst_y [dbget ${inst_ptr}.box_lly]
  set inst_orient [dbget ${inst_ptr}.orient]
  puts -nonewline $fp "${inst_name},${inst_master},${inst_width},${inst_height}"
  puts $fp ",${inst_x},${inst_y},${inst_orient}"
}

proc write_cluster_instance_details { cluster_id {output_dir "./"}} {
  set cluster_ptr [dbget top.fplan.groups.name ${cluster_id} -p]
  set fp [open ${output_dir}/${cluster_id}_inst.csv w]
  write_inst_header $fp
  
  foreach inst_ptr [dbget ${cluster_ptr}.members] {
    write_inst_details $fp $inst_ptr
  }
  close $fp
}

proc check_if_inst_in_cluster { cluster_id inst_ptr } {
  set cluster [dbget ${inst_ptr}.group.name]
  if {$cluster == $cluster_id} {
    return 1
  }
  return 0
}

proc get_net_source { net_ptr cluster_id {is_pin 0} } {
  if {$is_pin} {
    set inst_source [dbget [dbget ${net_ptr}.instTerms.isOutput 1 -p ].name -e]
  } else {
    set inst_source [dbget [dbget ${net_ptr}.instTerms.isOutput 1 -p \
        ].inst.name -e]
  }

  set inst_source [lindex $inst_source 0]
  set inst_ptr [dbget [dbget ${net_ptr}.instTerms.isOutput 1 -p ].inst -e]
  
  if {[check_if_inst_in_cluster $cluster_id $inst_ptr]} {
    return $inst_source
  }
  
  return "dummy_node"
}

proc get_net_sink { net_ptr cluster_id {is_pin 0} } {
  set sinks {}

  ## Check if there are any term sinks ##
  if { [dbget ${net_ptr}.terms.direction output -p -e] != "" } {
    set sink_name "dummy_node"
    lappend sinks $sink_name
  }

  ## List inst sink belong to the cluster ##
  foreach input_inst_term_ptr [dbget ${net_ptr}.instTerms.isInput 1 -p -e] {
    if {$is_pin} {
      set sink_name [dbget ${input_inst_term_ptr}.name -e]
    } else {
      set sink_name [dbget ${input_inst_term_ptr}.inst.name -e]
    }
    set inst_ptr [dbget ${input_inst_term_ptr}.inst -e]
    if {[check_if_inst_in_cluster $cluster_id $inst_ptr]} {
      lappend sinks $sink_name
    } else {
      lappend sinks "dummy_node"
    }
  }

  return [lsort -unique $sinks]
}

proc write_net_helper { fp node_name dummy_id } {
  if {$node_name == "dummy_node"} {
    puts -nonewline $fp ",dummy_node_${dummy_id}"
    incr dummy_id
  } else {
    puts -nonewline $fp ",${node_name}"
  }
  return $dummy_id
}

proc list_net_nodes { source sinks } {
  if {$source == "dummy_node"} {
    set idx [lsearch -exact $sinks "dummy_node"]
    if {$idx != -1} {
      set sinks [lreplace $sinks $idx $idx]
    }
  }
  set list_node [linsert $sinks 0 $source]
}

proc wirte_net_details { fp cluster_id net_ptr dummy_id {is_pin 0}} {
  set net_name [dbget ${net_ptr}.name]
  puts -nonewline $fp "${net_name}"
  set source [get_net_source $net_ptr $cluster_id $is_pin]
  set sinks [get_net_sink $net_ptr $cluster_id $is_pin]
  
  set nodes [list_net_nodes $source $sinks]
  foreach node $nodes {
    set dummy_id [write_net_helper $fp $node $dummy_id]
  }
  puts $fp ""
  return $dummy_id
}

proc is_external_net { net_ptr cluster_id {is_pin 0} } {
  set source [get_net_source $net_ptr $cluster_id $is_pin]
  set sinks [get_net_sink $net_ptr $cluster_id $is_pin]
  if {$source == "dummy_node"} {
    return 1
  }
  foreach sink $sinks {
    if {$sink == "dummy_node"} {
      return 1
    }
  }
  return 0
}

proc report_cluster_dummy_rent_con { cluster_id {is_pin 0} } {
  set cluster_ptr [dbget top.fplan.groups.name ${cluster_id} -p]
  set net_ptrs [dbget [dbget ${cluster_ptr}.members.instTerms.net \
      -u].isPwrOrGnd 0 -p -e]
  set net_count [llength $net_ptrs]
  set external_net_count 0
  foreach net_ptr $net_ptrs {
    if {[is_external_net $net_ptr $cluster_id $is_pin]} {
      incr external_net_count
    }
  }
  set dummy_rent_con [expr ${net_count}*1.0/${external_net_count}]
  return $dummy_rent_con
} 

proc report_dummy_rent_con { {is_pin 0} } {
  foreach cluster [dbget top.fplan.groups.name -e] {
    set cluster_rent_cont [report_cluster_dummy_rent_con $cluster $is_pin]
    puts "Cluster ${cluster} dummy rent cont: ${cluster_rent_cont}"
  }
}

proc create_soft_guides { {threshold 0.0} } {
  set cluster_rent_cont 0.0
  foreach cluster [dbget top.fplan.groups.name -e] {
    if { $threshold != 0.0 } {
      set cluster_rent_cont [report_cluster_dummy_rent_con $cluster 0]
    }
    if { $cluster_rent_cont >= $threshold } {
      createSoftGuide $cluster
    } else {
      deleteInstGroup $cluster
    }
  }
}

proc write_cluster_net_detail { cluster_id {output_dir "./"}} {
  set fp [open ${output_dir}/${cluster_id}_net.csv w]
  set dummy_id 0
  set dummy_id_prev 0
  set net_ptrs ""
  set cluster_ptr [dbget top.fplan.groups.name ${cluster_id} -p]
  foreach net_ptr [dbget ${cluster_ptr}.members.instTerms.net -u] {
    if {[dbget ${net_ptr}.isPwrOrGnd] == 1} {
      continue
    }
    set dummy_id [wirte_net_details $fp $cluster_id $net_ptr $dummy_id 0]
    if {$dummy_id != $dummy_id_prev} {
      set dummy_id_prev $dummy_id
      lappend net_ptrs $net_ptr
    }
  }
  close $fp
  return $net_ptrs
}

proc write_cluster_net_term_header { fp } {
  puts $fp "net_name,term_name,pt_x,pt_y"
}

proc check_if_net_in_cluster { net_box cluster_box } {
  set net_llx [lindex $net_box 0]
  set net_lly [lindex $net_box 1]
  set net_urx [lindex $net_box 2]
  set net_ury [lindex $net_box 3]
  
  set cluster_llx [lindex $cluster_box 0]
  set cluster_lly [lindex $cluster_box 1]
  set cluster_urx [lindex $cluster_box 2]
  set cluster_ury [lindex $cluster_box 3]
  
  if {($net_llx >= $cluster_llx) && ($net_lly >= $cluster_lly) \
      && ($net_urx <= $cluster_urx) && ($net_ury <= $cluster_ury)} {
    return 1 
  }
  return 0
}

proc find_net_dummy_node_loc_helper { llx lly cluster_box } {
  set cluster_llx [lindex $cluster_box 0]
  set cluster_lly [lindex $cluster_box 1]
  set cluster_urx [lindex $cluster_box 2]
  set cluster_ury [lindex $cluster_box 3]

  set is_inside 0
  if {($llx >= $cluster_llx) && ($lly >= $cluster_lly) \
      && ($llx <= $cluster_urx) && ($lly <= $cluster_ury)} {
    set is_inside 1
  }

  if {$is_inside == 0} {
    return [list 0 0 0]
  }

  set ld [expr $llx - $cluster_llx]
  set rd [expr $cluster_urx - $llx]
  set td [expr $cluster_ury - $lly]
  set bd [expr $lly - $cluster_lly]
  set d_list [ list $ld $rd $td $bd ]
  set min_d [lindex [lsort -real $d_list] 0]
  if { $ld == $min_d } {
    return [list 1 $cluster_llx $lly]
  } elseif { $rd == $min_d } {
    return [list 1 $cluster_urx $lly]
  } elseif { $td == $min_d } {
    return [list 1 $llx $cluster_ury]
  } else {
    return [list 1 $llx $cluster_lly]
  }
}

proc find_net_dummy_node_loc { net_box cluster_box } {
  # if cluster is inside the net box
  if { [check_if_net_in_cluster $cluster_box $net_box] } {
    return [lrange $cluster_box 0 1]
  }

  set net_llx [lindex $net_box 0]
  set net_lly [lindex $net_box 1]
  set net_urx [lindex $net_box 2]
  set net_ury [lindex $net_box 3]

  set cluster_llx [lindex $cluster_box 0]
  set cluster_lly [lindex $cluster_box 1]
  set cluster_urx [lindex $cluster_box 2]
  set cluster_ury [lindex $cluster_box 3]

  set net_dnode [find_net_dummy_node_loc_helper $net_llx $net_lly $cluster_box]
  if { [lindex $net_dnode 0] == 1 } {
    return [lrange $net_dnode 1 end]
  }
  set net_dnode [find_net_dummy_node_loc_helper $net_urx $net_ury $cluster_box]
  if { [lindex $net_dnode 0] == 1 } {
    return [lrange $net_dnode 1 end]
  }
  set net_dnode [find_net_dummy_node_loc_helper $net_llx $net_ury $cluster_box]
  if { [lindex $net_dnode 0] == 1 } {
    return [lrange $net_dnode 1 end]
  }
  set net_dnode [find_net_dummy_node_loc_helper $net_urx $net_lly $cluster_box]
  if { [lindex $net_dnode 0] == 1 } {
    return [lrange $net_dnode 1 end]
  }

  # If net box overlaps with cluster box but no vertices of netbox is inside
  # the cluster box
  if { $net_llx >= $cluster_llx && $net_llx <= $cluster_urx } {
    if { [expr $cluster_lly - $net_lly] < [expr $net_ury - $cluster_ury] } {
      return [list $net_llx $cluster_lly]
    } else {
      return [list $net_llx $cluster_ury]
    }
  } elseif { $net_urx >= $cluster_llx && $net_urx <= $cluster_urx } {
    if { [expr $cluster_lly - $net_lly] < [expr $net_ury - $cluster_ury] } {
      return [list $net_urx $cluster_lly]
    } else {
      return [list $net_urx $cluster_ury]
    }
  } elseif { $net_lly >= $cluster_lly && $net_lly <= $cluster_ury } {
    if { [expr $cluster_llx - $net_llx] < [expr $net_urx - $cluster_urx] } {
      return [list $cluster_llx $net_lly]
    } else {
      return [list $cluster_urx $net_lly]
    }
  } elseif { $net_ury >= $cluster_lly && $net_ury <= $cluster_ury } {
    if { [expr $cluster_llx - $net_llx] < [expr $net_urx - $cluster_urx] } {
      return [list $cluster_llx $net_ury]
    } else {
      return [list $cluster_urx $net_ury]
    }
  }
}

proc write_net_term { fp cluster_box net_ptr dummy_id} {
  set net_name [dbget ${net_ptr}.name]
  set net_box [lindex [dbget ${net_ptr}.box] 0]
  
  set dummy_node_loc [find_net_dummy_node_loc $net_box $cluster_box]
  set llx [lindex $dummy_node_loc 0]
  set lly [lindex $dummy_node_loc 1]
  puts $fp "${net_name},dummy_node_${dummy_id},${llx},${lly}"
}

proc write_cluster_net_terms { cluster_id net_ptrs {output_dir "./"}} {
  set fp [open ${output_dir}/${cluster_id}_net_dummy_terms.csv w]
  set dummy_id 0
  set cluster_bbox [is_valid_cluster $cluster_id]
  
  # remove first element and last two elemetns of the cluster bbox
  set cluster_bbox [lreplace $cluster_bbox 0 0]
  set cluster_bbox [lreplace $cluster_bbox end-1 end]

  set cluster_ptr [dbget top.fplan.groups.name ${cluster_id} -p]
  foreach net_ptr $net_ptrs {
    write_net_term $fp $cluster_bbox $net_ptr $dummy_id
    incr dummy_id
  }
  close $fp
}

## Function to write the cluster details:
proc write_cluster_details { cluster_id {output_dir "./"}} {
  if {![file exists $output_dir]} {
    file mkdir $output_dir
  }

  write_cluster_instance_details $cluster_id $output_dir
  set net_ptr [write_cluster_net_detail $cluster_id $output_dir]
  write_cluster_net_terms $cluster_id $net_ptr $output_dir
}

proc write_cluster_details_all { {output_dir "./"} } {
  if {![file exists $output_dir]} {
    file mkdir $output_dir
  }
  foreach cluster_id [dbget top.fplan.groups.name -u] {
    set is_valid [is_valid_cluster $cluster_id]
    if { [lindex $is_valid 0] == 1 } {
      write_cluster_details $cluster_id $output_dir
    }
  }
}

proc write_data {} {
  set top_cell [dbget top.name]
  deleteAllInstGroups
  defIn ${top_cell}_cluster_louvain.def
  write_cluster_details_all "./louvain_clusters"

  deleteAllInstGroups
  defIn ${top_cell}_cluster_leiden.def
  write_cluster_details_all "./leiden_clusters"
}

///////////////////////////////////////////////////////////////////////////
//
// BSD 3-Clause License
//
// Copyright (c) 2022, The Regents of the University of California
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////////
// High-level description
// This is the interfaces for TritonPart
// It works in two ways:
// 1) default mode :  read a verilog netlist and extract the hypergraph based on
// netlist 2) classical mode : read the hypergraph file in hmetis format
///////////////////////////////////////////////////////////////////////////////
#include "TritonPart.h"

#include <omp.h>

#include <iostream>
#include <numeric>
#include <set>
#include <string>

#include "Coarsener.h"
#include "Hypergraph.h"
#include "Multilevel.h"
#include "Partitioner.h"
#include "Refiner.h"
#include "odb/db.h"
#include "sta/ArcDelayCalc.hh"
#include "sta/Bfs.hh"
#include "sta/Corner.hh"
#include "sta/DcalcAnalysisPt.hh"
#include "sta/ExceptionPath.hh"
#include "sta/FuncExpr.hh"
#include "sta/Graph.hh"
#include "sta/GraphDelayCalc.hh"
#include "sta/Liberty.hh"
#include "sta/Network.hh"
#include "sta/PathAnalysisPt.hh"
#include "sta/PathEnd.hh"
#include "sta/PathExpanded.hh"
#include "sta/PathRef.hh"
#include "sta/PatternMatch.hh"
#include "sta/PortDirection.hh"
#include "sta/Sdc.hh"
#include "sta/Search.hh"
#include "sta/SearchPred.hh"
#include "sta/Sequential.hh"
#include "sta/Sta.hh"
#include "sta/Units.hh"
#include "utl/Logger.h"

using utl::PAR;

namespace par {

// -----------------------------------------------------------------------------------
// Public functions
// -----------------------------------------------------------------------------------

TritonPart::TritonPart(ord::dbNetwork* network,
                       odb::dbDatabase* db,
                       sta::dbSta* sta,
                       utl::Logger* logger)
    : network_(network), db_(db), sta_(sta), logger_(logger)
{
}

void TritonPart::SetTimingParams(float net_timing_factor,
                                 float path_timing_factor,
                                 float path_snaking_factor,
                                 float timing_exp_factor,
                                 float extra_delay,
                                 bool guardband_flag)
{
  // the timing weight for cutting a hyperedge
  net_timing_factor_ = net_timing_factor;

  // the cost for cutting a critical timing path once. If a critical
  // path is cut by 3 times, the cost is defined as 3 *
  // path_timing_factor_
  path_timing_factor_ = path_timing_factor;

  // the cost of introducing a snaking timing path, see our paper for
  // detailed explanation of snaking timing paths
  path_snaking_factor_ = path_snaking_factor;

  timing_exp_factor_ = timing_exp_factor;  // exponential factor
  extra_delay_ = extra_delay;              // extra delay introduced by cut
  guardband_flag_ = guardband_flag;        // timing guardband_flag
}

void TritonPart::SetFineTuneParams(
    // coarsening related parameters
    int thr_coarsen_hyperedge_size_skip,
    int thr_coarsen_vertices,
    int thr_coarsen_hyperedges,
    float coarsening_ratio,
    int max_coarsen_iters,
    float adj_diff_ratio,
    int min_num_vertices_each_part,
    // initial partitioning related parameters
    int num_initial_solutions,
    int num_best_initial_solutions,
    // refinement related parameters
    int refiner_iters,
    int max_moves,
    float early_stop_ratio,
    int total_corking_passes,
    // vcycle related parameters
    bool v_cycle_flag,
    int max_num_vcycle,
    int num_coarsen_solutions,
    int num_vertices_threshold_ilp,
    int global_net_threshold)
{
  // coarsening related parameters (stop conditions)

  // if the size of a hyperedge is larger than
  // thr_coarsen_hyperedge_size_skip_, then we ignore this hyperedge
  // during coarsening
  thr_coarsen_hyperedge_size_skip_ = thr_coarsen_hyperedge_size_skip;

  // the minimum threshold of number of vertices in the coarsest
  // hypergraph
  //thr_coarsen_vertices_ = thr_coarsen_vertices;
  //
  thr_coarsen_vertices_ = 500;

  // the minimum threshold of number of hyperedges in the coarsest
  // hypergraph
  thr_coarsen_hyperedges_ = thr_coarsen_hyperedges;
  
  // the ratio of number of vertices of adjacent coarse hypergraphs
  coarsening_ratio_ = coarsening_ratio;

  // maxinum number of coarsening iterations
  max_coarsen_iters_ = max_coarsen_iters;

  // the ratio of number of vertices of adjacent coarse hypergraphs if
  // the ratio is less than adj_diff_ratio_, then stop coarsening
  adj_diff_ratio_ = adj_diff_ratio;

  // minimum number of vertices in each block for the partitioning
  // solution of the coareset hypergraph We achieve this by
  // controlling the maximum vertex weight during coarsening
  min_num_vertices_each_part_ = min_num_vertices_each_part;

  // initial partitioning related parameters

  // number of initial random solutions generated
  num_initial_solutions_ = num_initial_solutions;

  // number of best initial solutions used for stability
  num_best_initial_solutions_ = num_best_initial_solutions;

  // refinement related parameter

  refiner_iters_ = refiner_iters;  // refinement iterations

  // the allowed moves for each pass of FM or greedy refinement or the
  // number of vertices in an ILP instance
  max_moves_ = max_moves;

  // If the number of moved vertices exceeds
  // num_vertices_of_a_hypergraph times early_stop_ratio_, then exit
  // current FM pass.  This parameter is set based on the obersvation
  // that in a FM pass, most of the gains are achieved by first
  // several moves
  early_stop_ratio_ = early_stop_ratio;

  // the maximum level of traversing the buckets to solve the "corking
  // effect"
  total_corking_passes_ = total_corking_passes;

  // V-cycle related parameters
  v_cycle_flag_ = v_cycle_flag;
  max_num_vcycle_ = max_num_vcycle;  // maximum number of vcycles
  num_coarsen_solutions_ = num_coarsen_solutions;

  // ILP threshold
  num_vertices_threshold_ilp_ = num_vertices_threshold_ilp;

  // global net threshold
  global_net_threshold_ = global_net_threshold;
}

// The function for partitioning a hypergraph
// This is used for replacing hMETIS
// Key supports:
// (1) fixed vertices constraint in fixed_file
// (2) community attributes in community_file (This can be used to guide the
// partitioning process) (3) stay together attributes in group_file. (4)
// placement information is specified in placement file The format is that each
// line cooresponds to a group fixed vertices, community and placement
// attributes both follows the hMETIS format
void TritonPart::PartitionHypergraph(unsigned int num_parts_arg,
                                     float balance_constraint_arg,
                                     std::vector<float> base_balance_arg,
                                     unsigned int seed_arg,
                                     int vertex_dimension_arg,
                                     int hyperedge_dimension_arg,
                                     int placement_dimension_arg,
                                     const char* hypergraph_file_arg,
                                     const char* fixed_file_arg,
                                     const char* community_file_arg,
                                     const char* group_file_arg,
                                     const char* placement_file_arg)
{
  logger_->report("========================================");
  logger_->report("[STATUS] Starting TritonPart Partitioner");
  logger_->report("========================================");
  logger_->info(PAR, 167, "Partitioning parameters**** ");
  // Parameters
  num_parts_ = num_parts_arg;
  base_balance_ = base_balance_arg;
  ub_factor_ = balance_constraint_arg;
  seed_ = seed_arg;
  vertex_dimensions_ = vertex_dimension_arg;
  hyperedge_dimensions_ = hyperedge_dimension_arg;
  placement_dimensions_ = placement_dimension_arg;
  // local parameters
  std::string hypergraph_file = hypergraph_file_arg;
  std::string fixed_file = fixed_file_arg;
  std::string community_file = community_file_arg;
  std::string group_file = group_file_arg;
  std::string placement_file = placement_file_arg;
  // solution file
  std::string solution_file
      = hypergraph_file + std::string(".part.") + std::to_string(num_parts_);
  logger_->info(PAR, 2, "Number of partitions = {}", num_parts_);
  logger_->info(PAR, 3, "UBfactor = {}", ub_factor_);
  logger_->info(PAR, 4, "Seed = {}", seed_);
  logger_->info(PAR, 5, "Vertex dimensions = {}", vertex_dimensions_);
  logger_->info(PAR, 6, "Hyperedge dimensions = {}", hyperedge_dimensions_);
  logger_->info(PAR, 7, "Placement dimensions = {}", placement_dimensions_);
  logger_->info(PAR, 8, "Hypergraph file = {}", hypergraph_file);
  logger_->info(PAR, 9, "Solution file = {}", solution_file);
  logger_->info(PAR, 10, "Global net threshold = {}", global_net_threshold_);
  if (!fixed_file.empty()) {
    logger_->info(PAR, 11, "Fixed file  = {}", fixed_file);
  }
  if (!community_file.empty()) {
    logger_->info(PAR, 12, "Community file = {}", community_file);
  }
  if (!group_file.empty()) {
    logger_->info(PAR, 13, "Group file = {}", group_file);
  }
  if (!placement_file.empty()) {
    logger_->info(PAR, 14, "Placement file = {}", placement_file);
  }

  // set the random seed
  srand(seed_);  // set the random seed

  timing_aware_flag_ = false;
  logger_->warn(PAR,
                119,
                "Reset the timing_aware_flag to false. Timing-driven mode is "
                "not supported");

  // build hypergraph: read the basic hypergraph information and other
  // constraints
  ReadHypergraph(
      hypergraph_file, fixed_file, community_file, group_file, placement_file);

  // call the multilevel partitioner to partition hypergraph_
  // but the evaluation is the original_hypergraph_
  MultiLevelPartition();

  // write the solution in hmetis format
  std::ofstream solution_file_output;
  solution_file_output.open(solution_file);
  for (auto part_id : solution_) {
    solution_file_output << part_id << std::endl;
  }
  solution_file_output.close();

  // finish hypergraph partitioning
  logger_->report("===============================================");
  logger_->report("Exiting TritonPart");
}

// Top level interface
// The function for partitioning a hypergraph
// This is the main API for TritonPart
// Key supports:
// (1) fixed vertices constraint in fixed_file
// (2) community attributes in community_file (This can be used to guide the
// partitioning process) (3) stay together attributes in group_file. (4)
// timing-driven partitioning (5) fence-aware partitioning (6) placement-aware
// partitioning, placement information is extracted from OpenDB
void TritonPart::PartitionDesign(unsigned int num_parts_arg,
                                 float balance_constraint_arg,
                                 std::vector<float> base_balance_arg,
                                 unsigned int seed_arg,
                                 bool timing_aware_flag_arg,
                                 int top_n_arg,
                                 bool placement_flag_arg,
                                 bool fence_flag_arg,
                                 float fence_lx_arg,
                                 float fence_ly_arg,
                                 float fence_ux_arg,
                                 float fence_uy_arg,
                                 const char* fixed_file_arg,
                                 const char* community_file_arg,
                                 const char* group_file_arg,
                                 const char* solution_filename_arg)
{
  logger_->report("========================================");
  logger_->report("[STATUS] Starting TritonPart Partitioner");
  logger_->report("========================================");
  logger_->info(PAR, 168, "[INFO] Partitioning parameters**** ");
  const int dbu = db_->getTech()->getDbUnitsPerMicron();
  block_ = db_->getChip()->getBlock();
  // Parameters
  num_parts_ = num_parts_arg;
  ub_factor_ = balance_constraint_arg;
  base_balance_ = base_balance_arg;
  seed_ = seed_arg;
  vertex_dimensions_ = 1;  // for design partitioning, vertex weight is the area
                           // of the instance
  hyperedge_dimensions_
      = 1;  // for design partitioning, hyperedge weight is the connectivity
  timing_aware_flag_ = timing_aware_flag_arg;
  if (timing_aware_flag_ == false) {
    top_n_ = 0;  // timing driven flow is disabled
  } else {
    top_n_ = top_n_arg;  // extract the top_n critical timing paths
  }
  placement_flag_ = placement_flag_arg;
  if (placement_flag_ == false) {
    placement_dimensions_ = 0;  // no placement information
  } else {
    placement_dimensions_ = 2;  // 2D canvas
  }
  fence_flag_ = fence_flag_arg;
  fence_ = Rect(fence_lx_arg * dbu,
                fence_ly_arg * dbu,
                fence_ux_arg * dbu,
                fence_uy_arg * dbu);
  if (fence_flag_ == false || fence_.IsValid() == false) {
    fence_.Reset();
  }
  // local parameters
  std::string fixed_file = fixed_file_arg;
  std::string community_file = community_file_arg;
  std::string group_file = group_file_arg;
  std::string solution_file = solution_filename_arg;
  logger_->info(PAR, 102, "Number of partitions = {}", num_parts_);
  logger_->info(PAR, 16, "UBfactor = {}", ub_factor_);
  logger_->info(PAR, 17, "Seed = {}", seed_);
  logger_->info(PAR, 18, "Vertex dimensions = {}", vertex_dimensions_);
  logger_->info(PAR, 19, "Hyperedge dimensions = {}", hyperedge_dimensions_);
  logger_->info(PAR, 20, "Placement dimensions = {}", placement_dimensions_);
  logger_->info(PAR, 21, "Timing aware flag = {}", timing_aware_flag_);
  logger_->info(PAR, 103, "Guardband flag = {}", guardband_flag_);
  logger_->info(PAR, 23, "Global net threshold = {}", global_net_threshold_);
  logger_->info(PAR, 24, "Top {} critical timing paths are extracted.", top_n_);
  logger_->info(PAR, 25, "Fence aware flag = {}", fence_flag_);
  if (fence_flag_ == true) {
    logger_->info(PAR,
                  26,
                  "fence_lx = {}, fence_ly = {}, fence_ux = {}, fence_uy = {}",
                  fence_.lx / dbu,
                  fence_.ly / dbu,
                  fence_.ux / dbu,
                  fence_.uy / dbu);
  }
  if (!fixed_file.empty()) {
    logger_->info(PAR, 27, "Fixed file  = {}", fixed_file);
  }
  if (!community_file.empty()) {
    logger_->info(PAR, 28, "Community file = {}", community_file);
  }
  if (!group_file.empty()) {
    logger_->info(PAR, 29, "Group file = {}", group_file);
  }
  if (!solution_file.empty()) {
    logger_->info(PAR, 30, "Solution file = {}", solution_file);
  }

  // set the random seed
  srand(seed_);  // set the random seed

  // read the netlist from OpenDB
  // for IO port and insts (std cells and macros),
  // there is an attribute for vertex_id
  logger_->report("========================================");
  logger_->report("[STATUS] Reading netlist**** ");
  // if the fence_flag_ is true, only consider the instances within the fence
  ReadNetlist(fixed_file, community_file, group_file);
  logger_->report("[STATUS] Finish reading netlist****");
  logger_->report("[STATUS] Generating switching activity****");
  // GetSwitchingActivity();
  logger_->report("[STATUS] Finish generating switching activity****");
  logger_->report("[STATUS] Reading hierarchy**** ");
  BuildLogicalHierarchy();
  logger_->report("[STATUS] Finish reading hierarchy**** ");
  BuildDendogram();
  // print the number of vertices in the tree
  logger_->report("Tree vertex count = {}", htree_->GetNumVertices());
  logger_->report("[STATUS] Clustering based on hierarchy**** ");
  int opts = 1;
  //std::vector<int> htree_clusters = ClusterHtree(opts);
  std::vector<int> htree_clusters = ClusterDendogram();
  logger_->report("[STATUS] Finished clustering**** ");
  // find number of clusters in the htree
  // number of clusters = number of unique cluster ids
  //std::set<int> htree_clusters_set(htree_clusters.begin(),
  //                                 htree_clusters.end());
  //logger_->report("Number of clusters in the htree = {}",
  //                htree_clusters_set.size());
  community_attr_ = htree_clusters;

  MultiLevelPartition();

  // Write out the solution.
  // Format 1: write the clustered netlist in verilog directly
  for (auto term : block_->getBTerms()) {
    auto vertex_id_property = odb::dbIntProperty::find(term, "vertex_id");
    const int vertex_id = vertex_id_property->getValue();
    if (vertex_id == -1) {
      continue;  // This instance is not used
    }
    const int partition_id = solution_[vertex_id];
    if (auto property = odb::dbIntProperty::find(term, "partition_id")) {
      property->setValue(partition_id);
    } else {
      odb::dbIntProperty::create(term, "partition_id", partition_id);
    }
  }

  for (auto inst : block_->getInsts()) {
    auto vertex_id_property = odb::dbIntProperty::find(inst, "vertex_id");
    const int vertex_id = vertex_id_property->getValue();
    if (vertex_id == -1) {
      continue;  // This instance is not used
    }
    const int partition_id = solution_[vertex_id];
    if (auto property = odb::dbIntProperty::find(inst, "partition_id")) {
      property->setValue(partition_id);
    } else {
      odb::dbIntProperty::create(inst, "partition_id", partition_id);
    }
  }

  // Format 2: write the explicit solution
  // each line :  instance_name  partition_id
  if (!solution_file.empty()) {
    std::string solution_file_name = solution_file;
    if (fence_flag_ == true) {
      // if the fence_flag_ is set to true, we need to update the solution file
      // to reflect the fence
      std::stringstream str_ss;
      str_ss.setf(std::ios::fixed);
      str_ss.precision(3);
      str_ss << ".lx_" << fence_.lx / dbu;
      str_ss << ".ly_" << fence_.ly / dbu;
      str_ss << ".ux_" << fence_.ux / dbu;
      str_ss << ".uy_" << fence_.uy / dbu;
      solution_file_name = solution_file_name + str_ss.str();
    }
    logger_->info(
        PAR, 110, "Updated solution file name = {}", solution_file_name);
    std::ofstream file_output;
    file_output.open(solution_file_name);

    for (auto term : block_->getBTerms()) {
      if (auto property = odb::dbIntProperty::find(term, "partition_id")) {
        file_output << term->getName() << "  ";
        file_output << property->getValue() << "  ";
        file_output << std::endl;
      }
    }

    for (auto inst : block_->getInsts()) {
      if (auto property = odb::dbIntProperty::find(inst, "partition_id")) {
        file_output << inst->getName() << "  ";
        file_output << property->getValue() << "  ";
        file_output << std::endl;
      }
    }
    file_output.close();
  }

  logger_->report("===============================================");
  logger_->report("Exiting TritonPart");
}

void TritonPart::EvaluateHypergraphSolution(unsigned int num_parts_arg,
                                            float balance_constraint_arg,
                                            std::vector<float> base_balance_arg,
                                            int vertex_dimension_arg,
                                            int hyperedge_dimension_arg,
                                            const char* hypergraph_file_arg,
                                            const char* fixed_file_arg,
                                            const char* group_file_arg,
                                            const char* solution_file_arg)

{
  logger_->report("========================================");
  logger_->report("[STATUS] Starting Evaluating Hypergraph Solution");
  logger_->report("========================================");
  logger_->info(PAR, 169, "Partitioning parameters**** ");
  // Parameters
  num_parts_ = num_parts_arg;
  ub_factor_ = balance_constraint_arg;
  base_balance_ = base_balance_arg;
  seed_ = 0;  // use the default random seed (no meaning in this function)
  vertex_dimensions_ = vertex_dimension_arg;
  hyperedge_dimensions_ = hyperedge_dimension_arg;
  placement_dimensions_
      = 0;  // use the default value (no meaning in this function)
  // local parameters
  std::string hypergraph_file = hypergraph_file_arg;
  std::string fixed_file = fixed_file_arg;
  std::string group_file = group_file_arg;
  // solution file
  std::string solution_file = solution_file_arg;
  std::string community_file;  // no community file is used (no meaning in this
                               // function)
  std::string placement_file;  // no placement file is used (no meaning in this
                               // function)
  logger_->info(PAR, 31, "Number of partitions = {}", num_parts_);
  logger_->info(PAR, 32, "UBfactor = {}", ub_factor_);
  logger_->info(PAR, 33, "Seed = {}", seed_);
  logger_->info(PAR, 34, "Vertex dimensions = {}", vertex_dimensions_);
  logger_->info(PAR, 35, "Hyperedge dimensions = {}", hyperedge_dimensions_);
  logger_->info(PAR, 36, "Placement dimensions = {}", placement_dimensions_);
  logger_->info(PAR, 37, "Hypergraph file = {}", hypergraph_file);
  logger_->info(PAR, 38, "Solution file = {}", solution_file);
  if (!fixed_file.empty()) {
    logger_->info(PAR, 39, "Fixed file  = {}", fixed_file);
  }
  if (!community_file.empty()) {
    logger_->info(PAR, 40, "Community file = {}", community_file);
  }
  if (!group_file.empty()) {
    logger_->info(PAR, 41, "Group file = {}", group_file);
  }
  if (!placement_file.empty()) {
    logger_->info(PAR, 42, "Placement file = {}", placement_file);
  }

  int part_id = -1;
  std::ifstream solution_file_input(solution_file);
  if (!solution_file_input.is_open()) {
    logger_->error(
        PAR, 2511, "Can not open the solution file : {}", solution_file);
  }
  while (solution_file_input >> part_id) {
    solution_.push_back(part_id);
  }
  solution_file_input.close();

  // set the random seed
  srand(seed_);  // set the random seed

  timing_aware_flag_ = false;
  logger_->warn(PAR,
                120,
                "Reset the timing_aware_flag to false. Timing-driven mode is "
                "not supported");

  // build hypergraph: read the basic hypergraph information and other
  // constraints
  ReadHypergraph(
      hypergraph_file, fixed_file, community_file, group_file, placement_file);

  // check the base balance constraint
  if (static_cast<int>(base_balance_.size()) != num_parts_) {
    logger_->warn(PAR, 350, "no base balance is specified. Use default value.");
    base_balance_.clear();
    base_balance_.resize(num_parts_);
    std::fill(base_balance_.begin(), base_balance_.end(), 1.0 / num_parts_);
  }

  // check the weighting scheme
  if (static_cast<int>(e_wt_factors_.size()) != hyperedge_dimensions_) {
    logger_->warn(
        PAR,
        121,
        "no hyperedge weighting is specified. Use default value of 1.");
    e_wt_factors_.clear();
    e_wt_factors_.resize(hyperedge_dimensions_);
    std::fill(e_wt_factors_.begin(), e_wt_factors_.end(), 1.0);
  }
  logger_->info(PAR,
                43,
                "hyperedge weight factor : [ {} ]",
                GetVectorString(e_wt_factors_));

  if (static_cast<int>(v_wt_factors_.size()) != vertex_dimensions_) {
    logger_->warn(
        PAR, 124, "No vertex weighting is specified. Use default value of 1.");
    v_wt_factors_.clear();
    v_wt_factors_.resize(vertex_dimensions_);
    std::fill(v_wt_factors_.begin(), v_wt_factors_.end(), 1.0);
  }
  logger_->info(
      PAR, 44, "vertex weight factor : [ {} ]", GetVectorString(v_wt_factors_));

  if (static_cast<int>(placement_wt_factors_.size()) != placement_dimensions_) {
    if (placement_dimensions_ <= 0) {
      placement_wt_factors_.clear();
    } else {
      logger_->warn(
          PAR,
          125,
          "No placement weighting is specified. Use default value of 1.");
      placement_wt_factors_.clear();
      placement_wt_factors_.resize(placement_dimensions_);
      std::fill(
          placement_wt_factors_.begin(), placement_wt_factors_.end(), 1.0f);
    }
  }
  logger_->info(PAR,
                45,
                "placement weight factor : [ {} ]",
                GetVectorString(placement_wt_factors_));

  // following parameters are not used
  net_timing_factor_ = 0.0;
  path_timing_factor_ = 0.0;
  path_snaking_factor_ = 0.0;
  timing_exp_factor_ = 0.0;
  extra_delay_ = 0.0;

  // print all the weighting parameters
  logger_->info(PAR, 46, "net_timing_factor : {}", net_timing_factor_);
  logger_->info(PAR, 47, "path_timing_factor : {}", path_timing_factor_);
  logger_->info(PAR, 48, "path_snaking_factor : {}", path_snaking_factor_);
  logger_->info(PAR, 49, "timing_exp_factor : {}", timing_exp_factor_);
  logger_->info(PAR, 50, "extra_delay : {}", extra_delay_);

  // create the evaluator class
  auto evaluator = std::make_shared<GoldenEvaluator>(num_parts_,
                                                     // weight vectors
                                                     e_wt_factors_,
                                                     v_wt_factors_,
                                                     placement_wt_factors_,
                                                     // timing related weight
                                                     net_timing_factor_,
                                                     path_timing_factor_,
                                                     path_snaking_factor_,
                                                     timing_exp_factor_,
                                                     extra_delay_,
                                                     original_hypergraph_,
                                                     logger_);

  evaluator->ConstraintAndCutEvaluator(original_hypergraph_,
                                       solution_,
                                       ub_factor_,
                                       base_balance_,
                                       group_attr_,
                                       true);

  logger_->report("===============================================");
  logger_->report("Exiting Evaluating Hypergraph Solution");
}

// Function to evaluate the hypergraph partitioning solution
// This can be used to write the timing-weighted hypergraph
// and evaluate the solution.
// If the solution file is empty, then this function is to write the
// solution. If the solution file is not empty, then this function is to
// evaluate the solution without writing the hypergraph again This function is
// only used for testing
void TritonPart::EvaluatePartDesignSolution(
    unsigned int num_parts_arg,
    float balance_constraint_arg,
    std::vector<float> base_balance_arg,
    bool timing_aware_flag_arg,
    int top_n_arg,
    bool fence_flag_arg,
    float fence_lx_arg,
    float fence_ly_arg,
    float fence_ux_arg,
    float fence_uy_arg,
    const char* fixed_file_arg,
    const char* community_file_arg,
    const char* group_file_arg,
    const char* hypergraph_file_arg,
    const char* hypergraph_int_weight_file_arg,
    const char* solution_filename_arg)
{
  logger_->report("========================================");
  logger_->report("[STATUS] Starting TritonPart Partitioner");
  logger_->report("========================================");
  logger_->info(PAR, 170, "Partitioning parameters**** ");
  const int dbu = db_->getTech()->getDbUnitsPerMicron();
  block_ = db_->getChip()->getBlock();
  // Parameters
  num_parts_ = num_parts_arg;
  ub_factor_ = balance_constraint_arg;
  base_balance_ = base_balance_arg;
  seed_ = 0;               // This parameter is not used.  just a random value
  vertex_dimensions_ = 1;  // for design partitioning, vertex weight is the area
                           // of the instance
  hyperedge_dimensions_
      = 1;  // for design partitioning, hyperedge weight is the connectivity
  timing_aware_flag_ = timing_aware_flag_arg;
  if (timing_aware_flag_ == false) {
    top_n_ = 0;  // timing driven flow is disabled
  } else {
    top_n_ = top_n_arg;  // extract the top_n critical timing paths
  }
  placement_flag_ = false;  // We do not need this parameter here
  if (placement_flag_ == false) {
    placement_dimensions_ = 0;  // no placement information
  } else {
    placement_dimensions_ = 2;  // 2D canvas
  }
  fence_flag_ = fence_flag_arg;
  fence_ = Rect(fence_lx_arg * dbu,
                fence_ly_arg * dbu,
                fence_ux_arg * dbu,
                fence_uy_arg * dbu);
  if (fence_flag_ == false || fence_.IsValid() == false) {
    fence_.Reset();
  }
  // local parameters
  std::string fixed_file = fixed_file_arg;
  std::string community_file = community_file_arg;
  std::string group_file = group_file_arg;
  std::string solution_file = solution_filename_arg;
  std::string hypergraph_file = hypergraph_file_arg;
  std::string hypergraph_int_weight_file = hypergraph_int_weight_file_arg;
  logger_->info(PAR, 104, "Number of partitions = {}", num_parts_);
  logger_->info(PAR, 52, "UBfactor = {}", ub_factor_);
  logger_->info(PAR, 53, "Seed = {}", seed_);
  logger_->info(PAR, 54, "Vertex dimensions = {}", vertex_dimensions_);
  logger_->info(PAR, 55, "Hyperedge dimensions = {}", hyperedge_dimensions_);
  logger_->info(PAR, 56, "Placement dimensions = {}", placement_dimensions_);
  logger_->info(PAR, 57, "Guardband flag = {}", guardband_flag_);
  logger_->info(PAR, 58, "Timing aware flag = {}", timing_aware_flag_);
  logger_->info(PAR, 59, "Global net threshold = {}", global_net_threshold_);
  logger_->info(PAR, 60, "Top {} critical timing paths are extracted.", top_n_);
  logger_->info(PAR, 61, "Fence aware flag = {}", fence_flag_);
  if (fence_flag_ == true) {
    logger_->info(PAR,
                  62,
                  "fence_lx = {}, fence_ly = {}, fence_ux = {}, fence_uy = {}",
                  fence_.lx / dbu,
                  fence_.ly / dbu,
                  fence_.ux / dbu,
                  fence_.uy / dbu);
  }
  if (!fixed_file.empty()) {
    logger_->info(PAR, 63, "Fixed file  = {}", fixed_file);
  }
  if (!community_file.empty()) {
    logger_->info(PAR, 64, "Community file = {}", community_file);
  }
  if (!group_file.empty()) {
    logger_->info(PAR, 65, "Group file = {}", group_file);
  }
  if (!hypergraph_file.empty()) {
    logger_->info(PAR, 66, "Hypergraph file = {}", hypergraph_file);
  }
  if (!hypergraph_int_weight_file.empty()) {
    logger_->info(
        PAR, 67, "Hypergraph_int_weight_file = {}", hypergraph_int_weight_file);
  }
  if (!solution_file.empty()) {
    logger_->info(PAR, 68, "Solution file = {}", solution_file);
  }

  // set the random seed
  srand(seed_);  // set the random seed

  // read the netlist from OpenDB
  // for IO port and insts (std cells and macros),
  // there is an attribute for vertex_id
  logger_->report("========================================");
  logger_->report("[STATUS] Reading netlist**** ");
  // if the fence_flag_ is true, only consider the instances within the fence
  ReadNetlist(fixed_file, community_file, group_file);
  logger_->report("[STATUS] Finish reading netlist****");

  // check the base balance constraint
  if (static_cast<int>(base_balance_.size()) != num_parts_) {
    logger_->warn(PAR, 351, "no base balance is specified. Use default value.");
    base_balance_.clear();
    base_balance_.resize(num_parts_);
    std::fill(base_balance_.begin(), base_balance_.end(), 1.0 / num_parts_);
  }

  // check the weighting scheme
  if (static_cast<int>(e_wt_factors_.size()) != hyperedge_dimensions_) {
    logger_->warn(
        PAR,
        126,
        "No hyperedge weighting is specified. Use default value of 1.");
    e_wt_factors_.clear();
    e_wt_factors_.resize(hyperedge_dimensions_);
    std::fill(e_wt_factors_.begin(), e_wt_factors_.end(), 1.0);
  }
  logger_->info(PAR,
                69,
                "hyperedge weight factor : [ {} ]",
                GetVectorString(e_wt_factors_));

  if (static_cast<int>(v_wt_factors_.size()) != vertex_dimensions_) {
    logger_->warn(
        PAR, 127, "No vertex weighting is specified. Use default value of 1.");
    v_wt_factors_.clear();
    v_wt_factors_.resize(vertex_dimensions_);
    std::fill(v_wt_factors_.begin(), v_wt_factors_.end(), 1.0);
  }
  logger_->info(
      PAR, 70, "vertex weight factor : [ {} ]", GetVectorString(v_wt_factors_));

  if (static_cast<int>(placement_wt_factors_.size()) != placement_dimensions_) {
    if (placement_dimensions_ <= 0) {
      placement_wt_factors_.clear();
    } else {
      logger_->warn(
          PAR,
          128,
          "No placement weighting is specified. Use default value of 1.");
      placement_wt_factors_.clear();
      placement_wt_factors_.resize(placement_dimensions_);
      std::fill(
          placement_wt_factors_.begin(), placement_wt_factors_.end(), 1.0f);
    }
  }
  logger_->info(PAR,
                105,
                "placement weight factor : [ {} ]",
                GetVectorString(placement_wt_factors_));

  // print all the weighting parameters
  logger_->info(PAR, 106, "net_timing_factor : {}", net_timing_factor_);
  logger_->info(PAR, 107, "path_timing_factor : {}", path_timing_factor_);
  logger_->info(PAR, 108, "path_snaking_factor : {}", path_snaking_factor_);
  logger_->info(PAR, 75, "timing_exp_factor : {}", timing_exp_factor_);
  logger_->info(PAR, 76, "extra_delay : {}", extra_delay_);

  // create the evaluator class
  auto evaluator = std::make_shared<GoldenEvaluator>(num_parts_,
                                                     // weight vectors
                                                     e_wt_factors_,
                                                     v_wt_factors_,
                                                     placement_wt_factors_,
                                                     // timing related weight
                                                     net_timing_factor_,
                                                     path_timing_factor_,
                                                     path_snaking_factor_,
                                                     timing_exp_factor_,
                                                     extra_delay_,
                                                     original_hypergraph_,
                                                     logger_);

  evaluator->InitializeTiming(original_hypergraph_);

  if (hypergraph_file.empty() == false) {
    evaluator->WriteWeightedHypergraph(original_hypergraph_, hypergraph_file);
    logger_->report("Finish writing hypergraph");
  }

  // This is for hMETIS. hMETIS only accept integer weight
  if (hypergraph_int_weight_file.empty() == false) {
    evaluator->WriteIntWeightHypergraph(original_hypergraph_,
                                        hypergraph_int_weight_file);
    logger_->report("Finish writing integer weight hypergraph");
  }

  if (solution_file.empty() == false) {
    int part_id = -1;
    std::ifstream solution_file_input(solution_file);
    if (!solution_file_input.is_open()) {
      logger_->error(
          PAR, 2514, "Can not open the solution file : {}", solution_file);
    }
    while (solution_file_input >> part_id) {
      solution_.push_back(part_id);
    }
    solution_file_input.close();
    evaluator->ConstraintAndCutEvaluator(original_hypergraph_,
                                         solution_,
                                         ub_factor_,
                                         base_balance_,
                                         group_attr_,
                                         true);

    // generate the timing report
    if (timing_aware_flag_ == true) {
      logger_->report("[STATUS] Displaying timing path cuts statistics");
      PathStats path_stats
          = evaluator->GetTimingCuts(original_hypergraph_, solution_);
      evaluator->PrintPathStats(path_stats);
    }

    logger_->report("===============================================");
    logger_->report("Exiting TritonPart");
    return;
  }
}

// k-way partitioning used by Hier-RTLMP
std::vector<int> TritonPart::PartitionKWaySimpleMode(
    unsigned int num_parts_arg,
    float balance_constraint_arg,
    unsigned int seed_arg,
    const std::vector<std::vector<int>>& hyperedges,
    const std::vector<float>& vertex_weights,
    const std::vector<float>& hyperedge_weights)
{
  num_parts_ = num_parts_arg;
  ub_factor_ = balance_constraint_arg;
  seed_ = seed_arg;
  vertex_dimensions_ = 1;  // for design partitioning, vertex weight is the area
                           // of the instance
  hyperedge_dimensions_
      = 1;  // for design partitioning, hyperedge weight is the connectivity
  timing_aware_flag_ = false;
  placement_flag_ = false;
  placement_dimensions_ = 0;
  fence_flag_ = false;
  hyperedges_ = hyperedges;
  fixed_attr_.clear();
  community_attr_.clear();
  group_attr_.clear();

  // convert vertex and hyperedge weights
  for (const auto& weight : vertex_weights) {
    std::vector<float> v_wt{weight};
    vertex_weights_.push_back(v_wt);
  }

  for (const auto& weight : hyperedge_weights) {
    std::vector<float> e_wt{weight};
    hyperedge_weights_.push_back(e_wt);
  }

  // Build the original hypergraph first
  original_hypergraph_ = std::make_shared<Hypergraph>(vertex_dimensions_,
                                                      hyperedge_dimensions_,
                                                      placement_dimensions_,
                                                      hyperedges_,
                                                      vertex_weights_,
                                                      hyperedge_weights_,
                                                      fixed_attr_,
                                                      community_attr_,
                                                      placement_attr_,
                                                      logger_);

  // call the multilevel partitioner to partition hypergraph_
  // but the evaluation is the original_hypergraph_
  MultiLevelPartition();

  return solution_;
}

// --------------------------------------------------------------------------------------
// Private functions
// --------------------------------------------------------------------------------------

// for hypergraph partitioning
// Read hypergraph from input files and related constraint files
void TritonPart::ReadHypergraph(const std::string& hypergraph_file,
                                const std::string& fixed_file,
                                const std::string& community_file,
                                const std::string& group_file,
                                const std::string& placement_file)
{
  // read hypergraph file
  std::ifstream hypergraph_file_input(hypergraph_file);
  if (!hypergraph_file_input.is_open()) {
    logger_->error(PAR,
                   2500,
                   "Can not open the input hypergraph file : {}",
                   hypergraph_file);
  }
  // Check the number of vertices, number of hyperedges, weight flag
  std::string cur_line;
  std::getline(hypergraph_file_input, cur_line);
  std::istringstream cur_line_buf(cur_line);
  std::vector<int> stats{std::istream_iterator<int>(cur_line_buf),
                         std::istream_iterator<int>()};
  num_hyperedges_ = stats[0];
  num_vertices_ = stats[1];
  bool hyperedge_weight_flag = false;
  bool vertex_weight_flag = false;
  if (stats.size() == 3) {
    if ((stats[2] % 10) == 1) {
      hyperedge_weight_flag = true;
    }
    if (stats[2] >= 10) {
      vertex_weight_flag = true;
    }
  }

  // clear the related vectors
  hyperedges_.clear();
  hyperedge_weights_.clear();
  vertex_weights_.clear();
  hyperedges_.reserve(num_hyperedges_);
  hyperedge_weights_.reserve(num_hyperedges_);
  vertex_weights_.reserve(num_vertices_);

  // Read hyperedge information
  for (int i = 0; i < num_hyperedges_; i++) {
    std::getline(hypergraph_file_input, cur_line);
    if (hyperedge_weight_flag == true) {
      std::istringstream cur_line_buf(cur_line);
      std::vector<float> hvec{std::istream_iterator<float>(cur_line_buf),
                              std::istream_iterator<float>()};
      std::vector<float>::iterator breakpoint{hvec.begin()
                                              + hyperedge_dimensions_};
      // read first hyperedge_dimensions_ elements as hyperege weights
      std::vector<float> hwts(hvec.begin(), breakpoint);
      // read remaining elements as hyperedge
      std::vector<int> hyperedge(breakpoint, hvec.end());
      for (auto& value : hyperedge) {
        value--;  // the vertex id starts from 1 in the hypergraph file
      }
      hyperedge_weights_.push_back(hwts);
      hyperedges_.push_back(hyperedge);
    } else {
      std::istringstream cur_line_buf(cur_line);
      std::vector<int> hyperedge{std::istream_iterator<int>(cur_line_buf),
                                 std::istream_iterator<int>()};
      for (auto& value : hyperedge) {
        value--;  // the vertex id starts from 1 in the hypergraph file
      }
      std::vector<float> hwts(hyperedge_dimensions_,
                              1.0);  // each dimension has the same weight
      hyperedge_weights_.push_back(hwts);
      hyperedges_.push_back(hyperedge);
    }
  }

  // Read weight for vertices
  for (int i = 0; i < num_vertices_; i++) {
    if (vertex_weight_flag == true) {
      std::getline(hypergraph_file_input, cur_line);
      std::istringstream cur_line_buf(cur_line);
      std::vector<float> vwts{std::istream_iterator<float>(cur_line_buf),
                              std::istream_iterator<float>()};
      vertex_weights_.push_back(vwts);
    } else {
      std::vector<float> vwts(vertex_dimensions_, 1.0);
      vertex_weights_.push_back(vwts);
    }
  }

  // Read fixed vertices
  if (!fixed_file.empty()) {
    int part_id = -1;
    std::ifstream fixed_file_input(fixed_file);
    if (!fixed_file_input.is_open()) {
      logger_->error(PAR, 2501, "Can not open the fixed file : {}", fixed_file);
    }
    while (fixed_file_input >> part_id) {
      fixed_attr_.push_back(part_id);
    }
    fixed_file_input.close();
    if (static_cast<int>(fixed_attr_.size()) != num_vertices_) {
      logger_->warn(PAR, 129, "Reset the fixed attributes to NONE.");
      fixed_attr_.clear();
    }
  }

  // Read community file
  if (!community_file.empty()) {
    int part_id = -1;
    std::ifstream community_file_input(community_file);
    if (!community_file_input.is_open()) {
      logger_->error(
          PAR, 2502, "Can not open the community file : {}", community_file);
    }
    while (community_file_input >> part_id) {
      community_attr_.push_back(part_id);
    }
    community_file_input.close();
    if (static_cast<int>(community_attr_.size()) != num_vertices_) {
      logger_->warn(PAR, 130, "Reset the community attributes to NONE.");
      community_attr_.clear();
    }
  }

  // read group file
  if (!group_file.empty()) {
    std::ifstream group_file_input(group_file);
    if (!group_file_input.is_open()) {
      logger_->error(PAR, 2503, "Can not open the group file : {}", group_file);
    }
    group_attr_.clear();
    std::string cur_line;
    while (std::getline(group_file_input, cur_line)) {
      std::istringstream cur_line_buf(cur_line);
      std::vector<int> group_info{std::istream_iterator<int>(cur_line_buf),
                                  std::istream_iterator<int>()};
      // reduce by 1 because definition of hMETIS
      for (auto& value : group_info) {
        value--;
      }
      if (group_info.size() > 1) {
        group_attr_.push_back(group_info);
      }
    }
    group_file_input.close();
  }

  // Read placement file
  if (!placement_file.empty()) {
    std::ifstream placement_file_input(placement_file);
    if (!placement_file_input.is_open()) {
      logger_->error(
          PAR, 2504, "Can not open the placement file : {}", placement_file);
    }
    std::string cur_line;
    // We assume the embedding has been normalized
    const float max_placement_value
        = 1.0;  // we assume the embedding has been normalized,
                // so we assume max_value is 1.0
    const float invalid_placement_thr
        = max_placement_value
          / 2.0;  // The threshold
                  // for detecting invalid placement value. If the abs(emb) is
                  // larger than invalid_placement_thr, we think the placement
                  // is invalid
    const float default_placement_value = 0.0;  // default placement value
    std::vector<std::vector<float>> temp_placement_attr;
    while (std::getline(placement_file_input, cur_line)) {
      std::vector<std::string> elements = SplitLine(
          cur_line);  // split the line based on deliminator empty space, ','
      std::vector<float> vertex_placement;
      for (auto& ele : elements) {
        if (ele == "NaN" || ele == "nan" || ele == "NAN") {
          vertex_placement.push_back(default_placement_value);
        } else {
          const float ele_value = std::stof(ele);
          if (std::abs(ele_value) < invalid_placement_thr) {
            vertex_placement.push_back(std::stof(ele));
          } else {
            vertex_placement.push_back(default_placement_value);
          }
        }
      }
      temp_placement_attr.push_back(vertex_placement);
    }
    placement_file_input.close();
    // Here comes the very important part for placement-driven clustering
    // Since we have so many vertices, the embedding value for each single
    // vertex usually very small, around e-5 - e-7 So we need to normalize the
    // placement embedding based on average distance again Here we randomly
    // sample num_vertices of pairs to compute the average norm
    std::vector<float> mean_placement_value_list(placement_dimensions_, 0.0f);
    for (auto& attr : temp_placement_attr) {
      for (int j = 0; j < placement_dimensions_; j++) {
        mean_placement_value_list[j] += attr[j];
      }
    }
    mean_placement_value_list = DivideFactor(mean_placement_value_list,
                                             temp_placement_attr.size() * 1.0);
    // perform normalization
    for (auto& emb : temp_placement_attr) {
      placement_attr_.push_back(
          DivideVectorElebyEle(emb, mean_placement_value_list));
    }

    if (static_cast<int>(placement_attr_.size()) != num_vertices_) {
      logger_->warn(PAR, 132, "Reset the placement attributes to NONE.");
      placement_attr_.clear();
    }
  }

  // Build the original hypergraph first
  original_hypergraph_ = std::make_shared<Hypergraph>(vertex_dimensions_,
                                                      hyperedge_dimensions_,
                                                      placement_dimensions_,
                                                      hyperedges_,
                                                      vertex_weights_,
                                                      hyperedge_weights_,
                                                      fixed_attr_,
                                                      community_attr_,
                                                      placement_attr_,
                                                      logger_);

  // show the status of hypergraph
  logger_->info(PAR, 171, "Hypergraph Information**");
  logger_->info(
      PAR, 172, "Vertices = {}", original_hypergraph_->GetNumVertices());
  logger_->info(
      PAR, 173, "Hyperedges = {}", original_hypergraph_->GetNumHyperedges());
}

// find the switching activity of each net in the netlist
void TritonPart::GetSwitchingActivity()
{
  switching_activity_.resize(original_hypergraph_->GetNumHyperedges());
  std::fill(switching_activity_.begin(), switching_activity_.end(), 0.0);
  float global_max_switching_activity = 0.0;
  for (auto net : block_->getNets()) {
    int hyperedge_id
        = odb::dbIntProperty::find(net, "hyperedge_id")->getValue();
    if (hyperedge_id == -1) {
      continue;  // the current net is not used
    }
    float max_switching_activity = 0.0;
    for (auto iterm : net->getITerms()) {
      // convert iterm to sta::Pin
      sta::Pin* pin = sta_->getDbNetwork()->dbToSta(iterm);
      odb::dbInst* inst = iterm->getInst();
      const int vertex_id
          = odb::dbIntProperty::find(inst, "vertex_id")->getValue();
      if (vertex_id == -1) {
        continue;  // the current instance is not used
      }
      sta::PwrActivity activity = sta_->findClkedActivity(pin);
      if (activity.isSet()) {
        max_switching_activity
            = std::max(max_switching_activity, activity.activity());
      }
    }
    switching_activity_[hyperedge_id] = max_switching_activity;
    global_max_switching_activity
        = std::max(global_max_switching_activity, max_switching_activity);
  }

  // normalize the switching activity to total switching activity
  for (int i = 0; i < original_hypergraph_->GetNumHyperedges(); i++) {
    float net_slack = hyperedge_slacks_[i];
    switching_activity_[i] /= global_max_switching_activity;
    float switch_cost = 1.0 + switching_activity_[i];
    // if net is critical then assign higher switching cost
    if (net_slack < 0.0) {
      // multiply the switching cost by the criticality value
      switch_cost *= (1.0 - net_slack);
    }
    switching_cost_.emplace_back(switch_cost);
  }

  // set the switching cost to the hypergraph
  original_hypergraph_->SetHyperedgeSwitchingCost(std::move(switching_cost_));
}

// for design partitioning
// read the logical hierarchy
// generate a logical hierarchy tree
// and generate clusters
void TritonPart::BuildLogicalHierarchy()
{
  // Get top module of the block (this is the root node of the hierarchy)
  odb::dbModule* top_module = block_->getTopModule();

  // Get all the instances in the top module
  std::vector<odb::dbInst*> insts;

  // Check if submodules exist
  if (top_module->getChildren().empty()) {
    // no logical hierarchy is found
    logger_->warn(PAR, 182, "No logical hierarchy is found");
    return;
  } else {
    logger_->info(PAR, 183, "Logical hierarchy is found");
    // create hierarchy nodes for the top module
    std::string module_name = top_module->getName();
    odb::dbSet<odb::dbModInst> children = top_module->getChildren();
    // convert dbSet to std::vector
    std::vector<odb::dbModInst*> children_modules;
    for (odb::dbModInst* child : children) {
      children_modules.push_back(child);
    }
    // check if some leaf instances exist here
    odb::dbSet<odb::dbInst> children_insts = top_module->getInsts();
    std::vector<odb::dbInst*> children_insts_vec;
    if (!children_insts.empty()) {
      for (odb::dbInst* inst : children_insts) {
        children_insts_vec.push_back(inst);
      }
    }
    // create a hierarchy node for the top module
    std::shared_ptr<HierarchyNode> top_node = std::make_shared<HierarchyNode>(
        module_name, children_modules, children_insts_vec);
    // First define the hierarchy tree
    htree_ = std::make_shared<HierarchyTree>();
    htree_->SetRoot(top_node);
    // Build the subtree from the root
    BuildSubTrees(top_node);
    // visit all nodes in the tree and assign vertex ids to them
    int tree_vtx_idx = 0;
    std::vector<std::shared_ptr<HierarchyNode>> nodes = htree_->GetNodes();
    for (std::shared_ptr<HierarchyNode> node : nodes) {
      node->SetVertexId(tree_vtx_idx);
      ++tree_vtx_idx;
    }
  }
}

void TritonPart::BuildSubTrees(HierNodePtr node)
{
  // Assert that node is not nullptr
  assert(node != nullptr);
  // Assert that tree is not nullptr
  assert(htree_ != nullptr);
  // Add the node to the tree
  htree_->AddNode(node);
  // Get the children modules of the current node
  std::vector<odb::dbModInst*> children = node->GetChildrenMods();

  while (true) {
    // Check if the current node has children
    if (children.empty()) {
      break;
    }
    // Iterate over all the children
    for (odb::dbModInst* child : children) {
      // Get the module of the child
      odb::dbModule* child_module = child->getMaster();
      // Get the name of the child module
      std::string child_module_name = child_module->getName();
      // Get the children modules of the child module
      odb::dbSet<odb::dbModInst> child_children = child_module->getChildren();
      // convert dbSet to std::vector
      std::vector<odb::dbModInst*> child_children_modules;
      for (odb::dbModInst* child_child : child_children) {
        child_children_modules.push_back(child_child);
      }
      // check if some leaf instances exist here
      odb::dbSet<odb::dbInst> child_leaf_insts = child_module->getInsts();
      std::vector<odb::dbInst*> child_leaf_insts_vec;
      if (!child_leaf_insts.empty()) {
        for (odb::dbInst* inst : child_leaf_insts) {
          child_leaf_insts_vec.push_back(inst);
        }
      }
      // create a hierarchy node for the child module
      std::shared_ptr<HierarchyNode> child_node
          = std::make_shared<HierarchyNode>(
              child_module_name, child_children_modules, child_leaf_insts_vec);
      // Add the child node to the current node
      node->AddChild(child_node);
      child_node->AddParent(node);
      // Build the subtree from the child node
      BuildSubTrees(child_node);
    }
    break;
  }
}

// Build a dendrogram based on the logical hierarchy tree
void TritonPart::BuildDendogram()
{
  // find how many levels are there in the hierarchy tree
  std::vector<int> levels_of_nodes(htree_->GetNumVertices(), 0);
  std::vector<HierNodePtr> nodes = htree_->GetNodes();
  std::vector<float> rent_params(nodes.size(), 0.0);
  // get root node
  HierNodePtr node = nodes[0];
  // set level of this node to 0
  levels_of_nodes[node->GetVertexId()] = 0;

  // define lambda function to get the subtree of a node
  std::function<std::vector<HierNodePtr>(HierNodePtr)> GetSubTree;

  GetSubTree = [&](HierNodePtr node) -> std::vector<HierNodePtr> {
    std::vector<HierNodePtr> subtree;
    subtree.push_back(node);
    std::vector<HierNodePtr> children = node->GetChildrenNodes();
    for (auto& child : children) {
      // subtree.push_back(child);
      std::vector<HierNodePtr> sub_subtree
          = GetSubTree(child);  // Now this call is valid
      subtree.insert(subtree.end(), sub_subtree.begin(), sub_subtree.end());
    }
    return subtree;
  };

  // define lambda to get all instances in a subtree
  auto GetSubTreeInsts = [&](std::vector<HierNodePtr> subtree) {
    std::vector<int> vertices_cluster;
    for (auto& node : subtree) {
      std::vector<odb::dbInst*> node_insts = node->GetChildrenInsts();
      // get the corresponding vertex ids of these instances
      for (auto& inst : node_insts) {
        int inst_id = odb::dbIntProperty::find(inst, "vertex_id")->getValue();
        vertices_cluster.push_back(inst_id);
      }
    }
    return vertices_cluster;
  };

  for (auto& node : nodes) {
    int node_id = node->GetVertexId();
    if (node_id == 0) {
      continue;
    }
    HierNodePtr parent = node->GetParent();
    int level_parent = levels_of_nodes[parent->GetVertexId()];
    levels_of_nodes[node->GetVertexId()] = level_parent + 1;
    // calculate the rent parameter of this cluster
    std::vector<HierNodePtr> subtree = GetSubTree(node);
    std::vector<int> vertices_cluster = GetSubTreeInsts(subtree);
    rent_params[node_id] = CalculateRentParam(vertices_cluster);
  }
  int max_level
      = *std::max_element(levels_of_nodes.begin(), levels_of_nodes.end());
  logger_->report("Max level = {}", max_level);
  // build a dendogram with max_level + 1 levels
  dendogram_ = std::make_shared<HierDendogram>(max_level);

  // traverse the hierarchy tree and build the dendogram datastructure based on
  // it
  std::vector<HierDendoNodePtr> dendogram_nodes;
  nodes = htree_->GetNodes();
  for (auto& node : nodes) {
    int level = levels_of_nodes[node->GetVertexId()];
    // get the instances in this node
    std::vector<odb::dbInst*> instances = node->GetChildrenInsts();
    // get the corresponding vertex ids of these instances
    std::vector<int> inst_ids;
    for (auto& inst : instances) {
      int inst_id = odb::dbIntProperty::find(inst, "vertex_id")->getValue();
      inst_ids.push_back(inst_id);
    }
    float rent_param = rent_params[node->GetVertexId()];
    // create a dendo node
    HierDendoNodePtr dendo_node = std::make_shared<HierDendogramNode>(
        inst_ids, rent_param, level, node->GetVertexId());
    dendogram_nodes.emplace_back(dendo_node);
  }

  int max_dendo_node_id = dendogram_nodes.size();

  // update children related information first
  for (auto& node : nodes) {
    // corresponding dendogram node
    int node_id = node->GetVertexId();
    // get the children of this node
    std::vector<HierNodePtr> children = node->GetChildrenNodes();
    // check if node has no children --> this is a leaf node
    if (children.size() == 0) {
      // find the level of this node
      int level = levels_of_nodes[node_id];
      if (level < max_level) {
        // replicate this node to be its child
        // create a dendo node and replicate this for max_level - level times
        for (int i = level; i < max_level; i++) {
          HierDendoNodePtr dendo_node = std::make_shared<HierDendogramNode>(
              dendogram_nodes[node_id]->GetVertexIds(),
              dendogram_nodes[node_id]->GetRentParam(),
              i + 1,
              max_dendo_node_id);
          dendogram_nodes.push_back(dendo_node);
          dendogram_nodes[node_id]->AddChild(dendo_node);
          node_id = max_dendo_node_id++;
        }
      }
    } else {
      // get the corresponding dendogram nodes of these children
      std::vector<HierDendoNodePtr> children_dendo_nodes;
      for (auto& child : children) {
        int child_id = child->GetVertexId();
        if (child_id == -1) {
          continue;
        }
        children_dendo_nodes.push_back(dendogram_nodes[child_id]);
      }
      for (auto& child : children_dendo_nodes) {
        dendogram_nodes[node_id]->AddChild(child);
      }
    }
  }

  // now update vertex ids for each dendo node
  // for example a parent node has 2 child node
  // child node 1 has vertex ids 1, 2, 3
  // child node 2 has vertex ids 4, 5, 6
  // then the parent node has vertex ids 1, 2, 3, 4, 5, 6
  // leaf nodes are already updated

  // print the level of all leaf nodes
  for (auto& node : dendogram_nodes) {
    if (node->GetNodeId() >= htree_->GetNumVertices()) {
      // calculate the rent parameter of this cluster
      std::vector<int> vertices_cluster = node->GetVertexIds();
      rent_params.push_back(CalculateRentParam(vertices_cluster));
    }
  }

  // traverse the dendogram from bottom to top
  for (int i = max_level - 1; i >= 0; i--) {
    // get all the nodes at level i
    std::vector<HierDendoNodePtr> nodes_at_level_i;
    for (auto& node : dendogram_nodes) {
      if (node->GetLevel() == i) {
        nodes_at_level_i.push_back(node);
      }
    }
    // update the vertex ids of these nodes
    for (auto& node : nodes_at_level_i) {
      // get the children of this node
      std::vector<HierDendoNodePtr> children = node->GetChildren();
      // get the vertex ids of these children
      std::vector<int> vertex_ids;
      for (auto& child : children) {
        std::vector<int> child_vertex_ids = child->GetVertexIds();
        vertex_ids.insert(
            vertex_ids.end(), child_vertex_ids.begin(), child_vertex_ids.end());
      }
      node->SetVertexIds(vertex_ids);
    }
  }

  dendogram_->SetNodes(dendogram_nodes);
  dendogram_->SetRoot(dendogram_nodes[0]);
  // set all rent parameters
  for (auto& node : dendogram_nodes) {
    dendogram_->SetNodeRentParam(node->GetNodeId(),
                                 rent_params[node->GetNodeId()]);
  }
}

std::vector<int> TritonPart::ClusterDendogram()
{
  int levels = dendogram_->GetMaxLevels();
  // try cluster from level 1 since level 0 is the entire netlist
  int min_lvl;
  float min_wt_avg_rent_param = std::numeric_limits<float>::max();
  std::vector<float> cluster_rent_param;
  for (int i = 1; i <= levels; ++i) {
    float wt_avg_rent_param = dendogram_->GetAvgRentAtLevel(i);
    cluster_rent_param.push_back(wt_avg_rent_param);
    if (wt_avg_rent_param < min_wt_avg_rent_param) {
      min_lvl = i;
      min_wt_avg_rent_param = wt_avg_rent_param;
    }
  }

  logger_->report("Clustering the hierarchy at level {} with rent parameter {}",
                  min_lvl,
                  min_wt_avg_rent_param);
  // get all nodes at level min_lvl
  std::vector<HierDendoNodePtr> nodes_at_min_lvl
      = dendogram_->GetNodesAtLevel(min_lvl);
  // assign vertices in each node to be in the same cluster
  std::vector<int> cluster_labels(original_hypergraph_->GetNumVertices(), -1);
  int cluster_id = 0;
  for (auto& node : nodes_at_min_lvl) {
    std::vector<int> vertex_ids = node->GetVertexIds();
    for (auto& vertex_id : vertex_ids) {
      cluster_labels[vertex_id] = cluster_id;
    }
    cluster_id++;
  }

  return cluster_labels;
}

// Cluster the htree_ to obtain clusters from the logical hierarchy
std::vector<int> TritonPart::ClusterHtree(int opts)
{
  // assert that htree_ is not nullptr
  assert(htree_ != nullptr);
  std::vector<int> cluster_labels(original_hypergraph_->GetNumVertices(), -1);
  // by default all the vertices are assigned to cluster -1
  // Loop through the nodes in the htree and assign all standard cell
  // instances to its parent node
  std::vector<HierNodePtr> nodes = htree_->GetNodes();
  std::vector<float> rents_param(nodes.size(), 0.0);
  for (auto& node : nodes) {
    // Check if node has leaf instances
    std::vector<odb::dbInst*> leaf_instances = node->GetChildrenInsts();
    // get id of this node
    int node_id = node->GetVertexId();
    std::vector<int> vertices_cluster;
    if (leaf_instances.size() > 0) {
      for (auto& inst : leaf_instances) {
        // find the vertex id of the instance
        int vertex_id = odb::dbIntProperty::find(inst, "vertex_id")->getValue();
        assert(cluster_labels[vertex_id] == -1);
        cluster_labels[vertex_id] = node_id;
        vertices_cluster.push_back(vertex_id);
      }
      // calculate the rent parameter of this cluster
      rents_param[node_id] = CalculateRentParam(vertices_cluster);
    }
  }

  // find average rent parameter of all leaf nodes
  float avg_rent_param = 0.0;
  float stdev_rent = 0.0;
  int freq = 0;
  for (auto& node : nodes) {
    if (node->GetChildrenInsts().size() > 0) {
      avg_rent_param += rents_param[node->GetVertexId()];
      freq++;
      stdev_rent += rents_param[node->GetVertexId()]
                    * rents_param[node->GetVertexId()];
    }
  }

  avg_rent_param = avg_rent_param / freq;
  stdev_rent = std::sqrt(stdev_rent / freq - avg_rent_param * avg_rent_param);

  logger_->info(PAR, 185, "Average rent parameter = {}", avg_rent_param);
  logger_->info(
      PAR, 186, "Standard deviation of rent parameter = {}", stdev_rent);

  // Loop through all clusters and try to refine clusters with poor rent
  // parameter
  std::set<int> unique_clusters
      = std::set<int>(cluster_labels.begin(), cluster_labels.end());
  std::vector<HierNodePtr> nodes_to_cluster;
  for (auto& cluster_label : unique_clusters) {
    if (cluster_label == -1) {
      continue;
    }
    // get the node from the htree
    HierNodePtr node = htree_->GetNode(cluster_label);
    // check the rent's parameter of the cluster
    float rent_param = rents_param[cluster_label];
    if (rent_param > (avg_rent_param + 1.5 * stdev_rent)) {
      // get the standard cell instances in this cluster
      std::vector<odb::dbInst*> leaf_instances = node->GetChildrenInsts();
      std::vector<int> vertices_cluster;
      for (auto& inst : leaf_instances) {
        // find the vertex id of the instance
        int vertex_id = odb::dbIntProperty::find(inst, "vertex_id")->getValue();
        vertices_cluster.push_back(vertex_id);
      }
      int best_cluster = -1;
      while (true) {
        if (node->GetVertexId() == 0) {
          break;
        }
        // get the parent of this node
        HierNodePtr parent = node->GetParent();
        // check if the parent is the root node
        if (parent->GetChildrenInsts().size() == 0) {
          node = parent;
          continue;
        } else {
          std::vector<odb::dbInst*> parent_instances
              = parent->GetChildrenInsts();
          for (auto& inst : parent_instances) {
            // find the vertex id of the instance
            int vertex_id
                = odb::dbIntProperty::find(inst, "vertex_id")->getValue();
            vertices_cluster.push_back(vertex_id);
          }
          // calculate the rent parameter of this cluster
          float new_rent_param = CalculateRentParam(vertices_cluster);
          if (new_rent_param <= rent_param) {
            best_cluster = parent->GetVertexId();
            break;
          } else {
            node = parent;
          }
        }
      }
      if (best_cluster > -1) {
        // assign all instances in vertices_cluster to parent cluster
        for (auto& vertex : vertices_cluster) {
          if (cluster_labels[vertex] != best_cluster) {
            cluster_labels[vertex] = best_cluster;
          }
          // collect all intances in child cluster
          std::vector<odb::dbInst*> child_instances = node->GetChildrenInsts();
          // add these as children to parent cluster
          for (auto& inst : child_instances) {
            HierNodePtr parent = htree_->GetNode(best_cluster);
            parent->AddChildrenInst(inst);
          }
          // remove these from child cluster
          node->RemoveChildrenInsts();
        }
      }
    }
  }

  // find the max cluster id
  int max_cluster_id
      = *std::max_element(cluster_labels.begin(), cluster_labels.end());

  // Assign each macro to be an individual cluster
  // Forcefully separate out the macros so that nothing clusters with them
  for (int i = 0; i < cluster_labels.size(); i++) {
    if (vertex_types_[i] == MACRO) {
      cluster_labels[i] = ++max_cluster_id;
    }
  }

  // Make the cluster labels continuous
  std::vector<int> cluster_labels_new(cluster_labels.size(), -1);
  int cluster_id = 0;
  for (auto& cluster_label : cluster_labels) {
    if (cluster_label == -1) {
      cluster_labels_new[cluster_id] = cluster_id;
    } else {
      cluster_labels_new[cluster_id] = cluster_label;
    }
    cluster_id++;
  }

  return cluster_labels_new;
}

float TritonPart::CalculateRentParam(std::vector<int>& vertex_ids)
{
  std::vector<int> labels(original_hypergraph_->GetNumVertices(), -1);
  for (int i = 0; i < vertex_ids.size(); i++) {
    labels[vertex_ids[i]] = 1;
  }
  // Find all hyperedges connected to vertices in the cluster
  std::set<int> hyperedges;
  for (auto& vertex : vertex_ids) {
    auto he = original_hypergraph_->Edges(vertex);
    hyperedges.insert(he.begin(), he.end());
  }

  // Find total outgoing hyperedges in the cluster
  int total_outgoing_hyperedges = 0;
  for (auto& hyperedge : hyperedges) {
    // Find all vertices in the hyperedge
    auto vertices_in_hyperedge = original_hypergraph_->Vertices(hyperedge);
    // Find all vertices in the hyperedge that connect to other clusters
    for (auto& vertex : vertices_in_hyperedge) {
      if (labels[vertex] == -1) {
        total_outgoing_hyperedges++;
        break;
      }
    }
  }

  float numerator = std::log(static_cast<float>(total_outgoing_hyperedges)
                             / static_cast<float>(hyperedges.size()));
  float denominator = std::log(static_cast<float>(vertex_ids.size()));
  if (denominator == 0) {
    return 1.0;
  } else {  // Return the rent parameter
    return 1.0 + (numerator / denominator);
  }
}

// for design partitioning
// Convert the netlist into hypergraphs
// read fixed_file, community_file and group_file
// read placement information
// read timing information
void TritonPart::ReadNetlist(const std::string& fixed_file,
                             const std::string& community_file,
                             const std::string& group_file)
{
  // assign vertex_id property of each instance and each IO port
  // the vertex_id property will be removed after the partitioning
  vertex_weights_.clear();
  vertex_types_.clear();
  fixed_attr_.clear();
  community_attr_.clear();
  group_attr_.clear();
  placement_attr_.clear();
  // traverse all the instances
  int vertex_id = 0;
  // check if the fence constraint is specified
  if (fence_flag_ == true) {
    // check IO ports
    for (auto term : block_->getBTerms()) {
      // -1 means that the instance is not used by the partitioner
      odb::dbIntProperty::create(term, "vertex_id", -1);
      odb::Rect box = term->getBBox();
      if (box.xMin() >= fence_.lx && box.xMax() <= fence_.ux
          && box.yMin() >= fence_.ly && box.yMax() <= fence_.uy) {
        odb::dbIntProperty::create(term, "vertex_id", vertex_id++);
        std::vector<float> vwts(vertex_dimensions_,
                                0.0);  // IO port has no area
        vertex_weights_.emplace_back(vwts);
        vertex_types_.emplace_back(PORT);
        vertex_names_.emplace_back(term->getName());
        odb::dbIntProperty::find(term, "vertex_id")->setValue(vertex_id++);
        if (placement_flag_ == true) {
          std::vector<float> loc{(box.xMin() + box.xMax()) / 2.0f,
                                 (box.yMin() + box.yMax()) / 2.0f};
          placement_attr_.emplace_back(loc);
        }
      }
    }
    // check instances
    for (auto inst : block_->getInsts()) {
      // get the name of the instance
      // -1 means that the instance is not used by the partitioner
      odb::dbIntProperty::create(inst, "vertex_id", -1);
      const sta::LibertyCell* liberty_cell = network_->libertyCell(inst);
      if (liberty_cell == nullptr) {
       // continue;  // ignore the instance with no liberty
      }
      odb::dbMaster* master = inst->getMaster();
      // get master name 
      std::string master_name = master->getName();
      // check if master name starts with "fake_ram"
      if (master->isPad() || master->isCover()) {
        continue;
      }
      odb::dbBox* box = inst->getBBox();
      // check if the inst is within the fence
      if (box->xMin() >= fence_.lx && box->xMax() <= fence_.ux
          && box->yMin() >= fence_.ly && box->yMax() <= fence_.uy) {
        const float area = liberty_cell->area();
        std::vector<float> vwts(vertex_dimensions_, area);
        vertex_weights_.emplace_back(vwts);
        if (master->isBlock()) {
          vertex_types_.emplace_back(MACRO);
        } else if (liberty_cell->hasSequentials()) {
          vertex_types_.emplace_back(SEQ_STD_CELL);
        } else {
          vertex_types_.emplace_back(COMB_STD_CELL);
        }
        vertex_names_.emplace_back(inst->getName());
        if (placement_flag_ == true) {
          std::vector<float> loc{(box->xMin() + box->xMax()) / 2.0f,
                                 (box->yMin() + box->yMax()) / 2.0f};
          placement_attr_.emplace_back(loc);
        }
        odb::dbIntProperty::find(inst, "vertex_id")->setValue(vertex_id++);
      }
    }
  } else {
    for (auto term : block_->getBTerms()) {
      odb::dbIntProperty::create(term, "vertex_id", vertex_id++);
      vertex_types_.emplace_back(PORT);
      vertex_names_.emplace_back(term->getName());
      std::vector<float> vwts(vertex_dimensions_, 0.0);
      vertex_weights_.push_back(vwts);
      if (placement_flag_ == true) {
        odb::Rect box = term->getBBox();
        std::vector<float> loc{(box.xMin() + box.xMax()) / 2.0f,
                               (box.yMin() + box.yMax()) / 2.0f};
        placement_attr_.emplace_back(loc);
      }
    }

    for (auto inst : block_->getInsts()) {
      // -1 means that the instance is not used by the partitioner
      odb::dbIntProperty::create(inst, "vertex_id", -1);
      const sta::LibertyCell* liberty_cell = network_->libertyCell(inst);
      if (liberty_cell == nullptr) {
        continue;  // ignore the instance with no liberty
      }
      odb::dbMaster* master = inst->getMaster();
      // check if the instance is a pad or a cover macro
      if (master->isPad() || master->isCover()) {
        continue;
      }
      const float area = liberty_cell->area();
      std::vector<float> vwts(vertex_dimensions_, area);
      vertex_weights_.emplace_back(vwts);
      if (master->isBlock()) {
        vertex_types_.emplace_back(MACRO);
      } else if (liberty_cell->hasSequentials()) {
        vertex_types_.emplace_back(SEQ_STD_CELL);
      } else {
        vertex_types_.emplace_back(COMB_STD_CELL);
      }
      vertex_names_.emplace_back(inst->getName());
      odb::dbIntProperty::find(inst, "vertex_id")->setValue(vertex_id++);
      if (placement_flag_ == true) {
        odb::dbBox* box = inst->getBBox();
        std::vector<float> loc{(box->xMin() + box->xMax()) / 2.0f,
                               (box->yMin() + box->yMax()) / 2.0f};
        placement_attr_.emplace_back(loc);
      }
    }
  }

  num_vertices_ = vertex_id;

  // read fixed instance file
  if (fixed_file.empty() == false) {
    std::ifstream file_input(fixed_file);
    if (!file_input.is_open()) {
      logger_->warn(
          PAR, 133, "Cannot open the fixed instance file : {}", fixed_file);
    } else {
      fixed_attr_.resize(num_vertices_);
      std::fill(fixed_attr_.begin(), fixed_attr_.end(), -1);
      std::string cur_line;
      while (std::getline(file_input, cur_line)) {
        std::stringstream ss(cur_line);
        std::string inst_name;
        int partition_id = -1;
        ss >> inst_name;
        ss >> partition_id;
        auto db_inst = block_->findInst(inst_name.c_str());
        const int vertex_id
            = odb::dbIntProperty::find(db_inst, "vertex_id")->getValue();
        if (vertex_id > -1) {
          fixed_attr_[vertex_id] = partition_id;
        }
      }
    }
    file_input.close();
  }

  // read community attribute file
  if (community_file.empty() == false) {
    std::ifstream file_input(community_file);
    if (!file_input.is_open()) {
      logger_->warn(
          PAR, 134, "Cannot open the community file : {}", community_file);
    } else {
      community_attr_.resize(num_vertices_);
      std::fill(community_attr_.begin(), community_attr_.end(), -1);
      std::string cur_line;
      while (std::getline(file_input, cur_line)) {
        std::stringstream ss(cur_line);
        std::string inst_name;
        int partition_id = -1;
        ss >> inst_name;
        ss >> partition_id;
        auto db_inst = block_->findInst(inst_name.c_str());
        const int vertex_id
            = odb::dbIntProperty::find(db_inst, "vertex_id")->getValue();
        if (vertex_id > -1) {
          community_attr_[vertex_id] = partition_id;
        }
      }
    }
    file_input.close();
  }

  // read the group file
  if (group_file.empty() == false) {
    std::ifstream file_input(group_file);
    if (!file_input.is_open()) {
      logger_->warn(PAR, 135, "Cannot open the group file : {}", group_file);
    } else {
      group_attr_.clear();
      std::string cur_line;
      while (std::getline(file_input, cur_line)) {
        std::stringstream ss(cur_line);
        std::string inst_name;
        std::vector<int> inst_group;
        while (ss >> inst_name) {
          auto db_inst = block_->findInst(inst_name.c_str());
          const int vertex_id
              = odb::dbIntProperty::find(db_inst, "vertex_id")->getValue();
          if (vertex_id > -1) {
            inst_group.push_back(vertex_id);
          }
        }
        if (inst_group.size() > 1) {
          group_attr_.push_back(inst_group);
        }
      }
    }
    file_input.close();
  }

  // Check all the hyperedges,
  // we do not check the parallel hyperedges
  // because we need to consider timing graph
  hyperedges_.clear();
  hyperedge_weights_.clear();
  // Each net correponds to an hyperedge
  // Traverse the hyperedge and assign hyperedge_id to each net
  // the hyperedge_id property will be removed after partitioning
  int hyperedge_id = 0;
  for (auto net : block_->getNets()) {
    odb::dbIntProperty::create(net, "hyperedge_id", -1);
    // ignore all the power net
    if (net->getSigType().isSupply()) {
      continue;
    }
    // check the hyperedge
    int driver_id = -1;      // vertex id of the driver instance
    std::set<int> loads_id;  // vertex id of sink instances
    // check the connected instances
    for (odb::dbITerm* iterm : net->getITerms()) {
      odb::dbInst* inst = iterm->getInst();
      const int vertex_id
          = odb::dbIntProperty::find(inst, "vertex_id")->getValue();
      if (vertex_id == -1) {
        continue;  // the current instance is not used
      }
      if (iterm->getIoType() == odb::dbIoType::OUTPUT) {
        driver_id = vertex_id;
      } else {
        loads_id.insert(vertex_id);
      }
    }
    // check the connected IO pins
    for (odb::dbBTerm* bterm : net->getBTerms()) {
      const int vertex_id
          = odb::dbIntProperty::find(bterm, "vertex_id")->getValue();
      if (vertex_id == -1) {
        continue;  // the current bterm is not used
      }
      if (bterm->getIoType() == odb::dbIoType::INPUT) {
        driver_id = vertex_id;
      } else {
        loads_id.insert(vertex_id);
      }
    }
    // check the hyperedges
    std::vector<int> hyperedge;
    if (driver_id != -1 && !loads_id.empty()) {
      hyperedge.push_back(driver_id);
      for (auto& load_id : loads_id) {
        if (load_id != driver_id) {
          hyperedge.push_back(load_id);
        }
      }
    }
    // Ignore all the single-vertex hyperedge and large global netthreshold
    // if (hyperedge.size() > 1 && hyperedge.size() <= global_net_threshold_)
    // {
    if (hyperedge.size() > 1) {
      hyperedges_.push_back(hyperedge);
      hyperedge_weights_.emplace_back(hyperedge_dimensions_, 1.0);
      odb::dbIntProperty::find(net, "hyperedge_id")->setValue(hyperedge_id++);
    }
  }  // finish hyperedge
  num_hyperedges_ = static_cast<int>(hyperedges_.size());

  // add timing features
  if (timing_aware_flag_ == true) {
    logger_->report("[STATUS] Extracting timing paths**** ");
    BuildTimingPaths();  // create timing paths
  }

  if (num_vertices_ == 0 || num_hyperedges_ == 0) {
    logger_->error(PAR, 2677, "There is no vertices and hyperedges");
  }

  // build the timing graph
  // map each net to the timing arc in the timing graph
  std::vector<std::set<int>> hyperedges_arc_set;
  for (int e = 0; e < num_hyperedges_; e++) {
    const std::set<int> arc_set{e};
    hyperedges_arc_set.push_back(arc_set);
  }

  original_hypergraph_ = std::make_shared<Hypergraph>(vertex_dimensions_,
                                                      hyperedge_dimensions_,
                                                      placement_dimensions_,
                                                      hyperedges_,
                                                      vertex_weights_,
                                                      hyperedge_weights_,
                                                      fixed_attr_,
                                                      community_attr_,
                                                      placement_attr_,
                                                      vertex_types_,
                                                      hyperedge_slacks_,
                                                      hyperedges_arc_set,
                                                      timing_paths_,
                                                      logger_);

  original_hypergraph_->SetVertexNames(vertex_names_);

  std::string design_name = block_->getName();
  // write a file out for mapping of names to vertex ids
  std::ofstream file_output(design_name + ".names");
  if (!file_output.is_open()) {
    logger_->warn(PAR, 184, "Cannot open the vertex names file");
  } else {
    file_output << "Name,VertexId\n";
    int vertex_id = 0;
    for (auto& name : vertex_names_) {
      file_output << name << "," << vertex_id++ << "\n";
    }
  }

  // write out a hypergraph file 
  // first line is number of hyperedges and number of vertices
  // line i+1 is the list of vertices in hyperedge i

  std::ofstream file_output2(design_name + ".hgr");
  if (!file_output2.is_open()) {
    logger_->warn(PAR, 187, "Cannot open the hypergraph file");
  } else {
    file_output2 << num_hyperedges_ << " " << num_vertices_ << "\n";
    for (auto& hyperedge : hyperedges_) {
      for (auto& vertex : hyperedge) {
        file_output2 << vertex << " ";
      }
      file_output2 << "\n";
    }
  }

  // show the status of hypergraph
  logger_->info(PAR, 174, "Netlist Information**");
  logger_->info(
      PAR, 175, "Vertices = {}", original_hypergraph_->GetNumVertices());
  logger_->info(
      PAR, 176, "Hyperedges = {}", original_hypergraph_->GetNumHyperedges());
  logger_->info(PAR, 177, "Number of timing paths = {}", timing_paths_.size());
}

// Find all the critical timing paths
// The codes below similar to gui/src/staGui.cpp
// Please refer to sta/Search/ReportPath.cc for how to check the timing path
// Currently we can only consider single-clock design
// TODO:  how to handle multi-clock design
void TritonPart::BuildTimingPaths()
{
  if (timing_aware_flag_ == false || top_n_ <= 0) {
    logger_->warn(PAR, 136, "Timing driven partitioning is disabled");
    return;
  }
  sta_->ensureGraph();     // Ensure that the timing graph has been built
  sta_->searchPreamble();  // Make graph and find delays
  sta_->ensureLevelized();
  // Step 1:  find the top_n critical timing paths
  sta::ExceptionFrom* e_from = nullptr;
  sta::ExceptionThruSeq* e_thrus = nullptr;
  sta::ExceptionTo* e_to = nullptr;
  bool include_unconstrained = false;
  bool get_max = true;  // max for setup check, min for hold check
  // Timing paths are grouped into path groups according to the clock
  // associated with the endpoint of the path, for example, path group for clk
  int group_count = top_n_;
  int endpoint_count = 1;  // The number of paths to report for each endpoint.
  // Definition for findPathEnds function in Search.hh
  // PathEndSeq *findPathEnds(ExceptionFrom *from,
  //              ExceptionThruSeq *thrus,
  //              ExceptionTo *to,
  //              bool unconstrained,
  //              const Corner *corner,
  //              const MinMaxAll *min_max,
  //              int group_count,
  //              int endpoint_count,
  //              bool unique_pins,
  //              float slack_min,
  //              float slack_max,
  //              bool sort_by_slack,
  //              PathGroupNameSet *group_names,
  //              bool setup,
  //              bool hold,
  //              bool recovery,
  //              bool removal,
  //              bool clk_gating_setup,
  //              bool clk_gating_hold);
  // PathEnds represent search endpoints that are either unconstrained or
  // constrained by a timing check, output delay, data check, or path delay.
  sta::PathEndSeq path_ends = sta_->search()->findPathEnds(  // from, thrus,
                                                             // to,
                                                             // unconstrained
      e_from,   // return paths from a list of clocks/instances/ports/register
                // clock pins or latch data pins
      e_thrus,  // return paths through a list of instances/ports/nets
      e_to,     // return paths to a list of clocks/instances/ports or pins
      include_unconstrained,  // return unconstrained paths
      // corner, min_max,
      sta_->cmdCorner(),  // return paths for a process corner
      get_max ? sta::MinMaxAll::max()
              : sta::MinMaxAll::min(),  // return max/min paths checks
      // group_count, endpoint_count, unique_pins
      group_count,     // number of paths in total
      endpoint_count,  // number of paths for each endpoint
      true,
      -sta::INF,
      sta::INF,  // slack_min, slack_max,
      true,      // sort_by_slack
      nullptr,   // group_names
      // setup, hold, recovery, removal,
      get_max,
      !get_max,
      false,
      false,
      // clk_gating_setup, clk_gating_hold
      false,
      false);

  // check all the timing paths
  for (auto& path_end : path_ends) {
    // Printing timing paths to logger
    // sta_->reportPathEnd(path_end);
    auto* path = path_end->path();
    TimingPath timing_path;                     // create the timing path
    const float slack = path_end->slack(sta_);  // slack information
    // TODO: to be deleted.  We should not
    // normalize the slack according to the clock period for multi-clock
    // design
    const float clock_period = path_end->targetClk(sta_)->period();
    maximum_clock_period_ = std::max(maximum_clock_period_, clock_period);
    timing_path.slack = slack;
    // logger_->report("clock_period = {}, slack = {}", maximum_clock_period_,
    // timing_path.slack);
    sta::PathExpanded expand(path, sta_);
    expand.path(expand.size() - 1);
    for (size_t i = 0; i < expand.size(); i++) {
      // PathRef is reference to a path vertex
      sta::PathRef* ref = expand.path(i);
      sta::Pin* pin = ref->vertex(sta_)->pin();
      // Nets connect pins at a level of the hierarchy
      auto net = network_->net(pin);  // sta::Net*
      // Check if the pin is connected to a net
      if (net == nullptr) {
        continue;  // check if the net exists
      }
      if (network_->isTopLevelPort(pin) == true) {
        auto bterm = block_->findBTerm(network_->pathName(pin));
        const int vertex_id
            = odb::dbIntProperty::find(bterm, "vertex_id")->getValue();
        if (vertex_id == -1) {
          continue;
        }
        if (timing_path.path.empty() == true
            || timing_path.path.back() != vertex_id) {
          timing_path.path.push_back(vertex_id);
        }
      } else {
        auto inst = network_->instance(pin);
        auto db_inst = block_->findInst(network_->pathName(inst));
        const int vertex_id
            = odb::dbIntProperty::find(db_inst, "vertex_id")->getValue();
        if (vertex_id == -1) {
          continue;
        }
        if (timing_path.path.empty() == true
            || timing_path.path.back() != vertex_id) {
          timing_path.path.push_back(vertex_id);
        }
      }
      auto db_net = block_->findNet(
          network_->pathName(net));  // convert sta::Net* to dbNet*
      const int hyperedge_id
          = odb::dbIntProperty::find(db_net, "hyperedge_id")->getValue();
      if (hyperedge_id == -1) {
        continue;
      }
      timing_path.arcs.push_back(hyperedge_id);
    }
    // add timing path
    if (!timing_path.arcs.empty()) {
      timing_paths_.push_back(timing_path);
    }
  }

  // normalize all the slack
  logger_->info(
      PAR, 178, "maximum_clock_period : {} second", maximum_clock_period_);
  extra_delay_ = extra_delay_ / maximum_clock_period_;
  logger_->info(PAR, 179, "normalized extra delay : {}", extra_delay_);
  if (guardband_flag_ == false) {
    for (auto& timing_path : timing_paths_) {
      timing_path.slack = timing_path.slack / maximum_clock_period_;
    }
  } else {
    for (auto& timing_path : timing_paths_) {
      timing_path.slack
          = timing_path.slack / maximum_clock_period_ - extra_delay_;
    }
  }
  logger_->info(PAR,
                180,
                "We normalized the slack of each path based on maximum clock "
                "period");
  // resize the hyperedge_slacks_
  hyperedge_slacks_.clear();
  hyperedge_slacks_.resize(num_hyperedges_);
  std::fill(hyperedge_slacks_.begin(),
            hyperedge_slacks_.end(),
            maximum_clock_period_);
  logger_->info(
      PAR,
      181,
      "We normalized the slack of each net based on maximum clock period");
  int num_unconstrained_hyperedges = 0;
  // check the slack on each net
  for (auto db_net : block_->getNets()) {
    const int hyperedge_id
        = odb::dbIntProperty::find(db_net, "hyperedge_id")->getValue();
    if (hyperedge_id == -1) {
      continue;  // this net is not used
    }
    sta::Net* net = network_->dbToSta(db_net);
    const float slack = sta_->netSlack(net, sta::MinMax::max());
    // set the slack of unconstrained net to max_clock_period_
    if (slack > maximum_clock_period_) {
      num_unconstrained_hyperedges++;
      hyperedge_slacks_[hyperedge_id] = 1.0;
    } else {
      if (guardband_flag_ == false) {
        hyperedge_slacks_[hyperedge_id] = slack / maximum_clock_period_;
      } else {
        hyperedge_slacks_[hyperedge_id]
            = slack / maximum_clock_period_ - extra_delay_;
      }
    }
  }
  logger_->report("[STATUS] Finish traversing timing graph");
  if (num_unconstrained_hyperedges > 0) {
    logger_->warn(PAR,
                  137,
                  "{} unconstrained hyperedges !",
                  num_unconstrained_hyperedges);
  }
  logger_->warn(PAR,
                138,
                "Reset the slack of all unconstrained hyperedges to {} seconds",
                maximum_clock_period_);
}

// Partition the hypergraph_ with the multilevel methodology
// the return value is the partitioning solution
void TritonPart::MultiLevelPartition()
{
  auto start_time_stamp_global = std::chrono::high_resolution_clock::now();

  // check the base balance constraint
  if (static_cast<int>(base_balance_.size()) != num_parts_) {
    logger_->warn(PAR, 352, "no base balance is specified. Use default value.");
    base_balance_.clear();
    base_balance_.resize(num_parts_);
    std::fill(base_balance_.begin(), base_balance_.end(), 1.0 / num_parts_);
  }

  // check the weighting scheme
  if (static_cast<int>(e_wt_factors_.size()) != hyperedge_dimensions_) {
    logger_->warn(
        PAR,
        139,
        "No hyperedge weighting is specified. Use default value of 1.");
    e_wt_factors_.clear();
    e_wt_factors_.resize(hyperedge_dimensions_);
    std::fill(e_wt_factors_.begin(), e_wt_factors_.end(), 1.0);
  }
  logger_->info(PAR,
                77,
                "hyperedge weight factor : [ {} ]",
                GetVectorString(e_wt_factors_));

  if (static_cast<int>(v_wt_factors_.size()) != vertex_dimensions_) {
    logger_->warn(
        PAR, 141, "No vertex weighting is specified. Use default value of 1.");
    v_wt_factors_.clear();
    v_wt_factors_.resize(vertex_dimensions_);
    std::fill(v_wt_factors_.begin(), v_wt_factors_.end(), 1.0);
  }
  logger_->info(
      PAR, 78, "vertex weight factor : [ {} ]", GetVectorString(v_wt_factors_));

  if (static_cast<int>(placement_wt_factors_.size()) != placement_dimensions_) {
    if (placement_dimensions_ <= 0) {
      placement_wt_factors_.clear();
    } else {
      logger_->warn(
          PAR,
          140,
          "No placement weighting is specified. Use default value of 1.");
      placement_wt_factors_.clear();
      placement_wt_factors_.resize(placement_dimensions_);
      std::fill(
          placement_wt_factors_.begin(), placement_wt_factors_.end(), 1.0f);
    }
  }
  logger_->info(PAR,
                79,
                "placement weight factor : [ {} ]",
                GetVectorString(placement_wt_factors_));
  // print all the weighting parameters
  logger_->info(PAR, 80, "net_timing_factor : {}", net_timing_factor_);
  logger_->info(PAR, 81, "path_timing_factor : {}", path_timing_factor_);
  logger_->info(PAR, 82, "path_snaking_factor : {}", path_snaking_factor_);
  logger_->info(PAR, 83, "timing_exp_factor : {}", timing_exp_factor_);
  // coarsening related parameters
  logger_->info(PAR, 84, "coarsen order : {}", ToString(coarsen_order_));
  logger_->info(PAR,
                85,
                "thr_coarsen_hyperedge_size_skip : {}",
                thr_coarsen_hyperedge_size_skip_);
  logger_->info(PAR, 86, "thr_coarsen_vertices : {}", thr_coarsen_vertices_);
  logger_->info(
      PAR, 87, "thr_coarsen_hyperedges : {}", thr_coarsen_hyperedges_);
  logger_->info(PAR, 88, "coarsening_ratio : {}", coarsening_ratio_);
  logger_->info(PAR, 89, "max_coarsen_iters : {}", max_coarsen_iters_);
  logger_->info(PAR, 90, "adj_diff_ratio : {}", adj_diff_ratio_);
  logger_->info(
      PAR, 91, "min_num_vertcies_each_part : {}", min_num_vertices_each_part_);
  // initial partitioning parameter
  logger_->info(PAR, 92, "num_initial_solutions : {}", num_initial_solutions_);
  logger_->info(
      PAR, 93, "num_best_initial_solutions : {}", num_best_initial_solutions_);
  // refinement related parameters
  logger_->info(PAR, 94, "refine_iters : {}", refiner_iters_);
  logger_->info(
      PAR, 95, "max_moves (FM or greedy refinement) : {}", max_moves_);
  logger_->info(PAR, 96, "early_stop_ratio : {}", early_stop_ratio_);
  logger_->info(PAR, 97, "total_corking_passes : {}", total_corking_passes_);
  logger_->info(PAR, 98, "v_cycle_flag : {}", v_cycle_flag_);
  logger_->info(PAR, 99, "max_num_vcycle : {}", max_num_vcycle_);
  logger_->info(PAR, 100, "num_coarsen_solutions : {}", num_coarsen_solutions_);
  logger_->info(
      PAR, 101, "num_vertices_threshold_ilp : {}", num_vertices_threshold_ilp_);

  // create the evaluator class
  auto tritonpart_evaluator
      = std::make_shared<GoldenEvaluator>(num_parts_,
                                          // weight vectors
                                          e_wt_factors_,
                                          v_wt_factors_,
                                          placement_wt_factors_,
                                          // timing related weight
                                          net_timing_factor_,
                                          path_timing_factor_,
                                          path_snaking_factor_,
                                          timing_exp_factor_,
                                          extra_delay_,
                                          original_hypergraph_,
                                          logger_);

  Matrix<float> upper_block_balance
      = original_hypergraph_->GetUpperVertexBalance(
          num_parts_, ub_factor_, base_balance_);

  Matrix<float> lower_block_balance
      = original_hypergraph_->GetLowerVertexBalance(
          num_parts_, ub_factor_, base_balance_);

  // Step 1 : create all the coarsening, partitionig and refinement class
  // TODO:  This may need to modify as lower_block_balance
  /*const std::vector<float> thr_cluster_weight
      = DivideFactor(original_hypergraph_->GetTotalVertexWeights(),
                     min_num_vertices_each_part_ * num_parts_);*/
  // manually set this value to 2000
  const std::vector<float>
  thr_cluster_weight(original_hypergraph_->GetVertexDimensions(), 500.0);

  logger_->report("[STATUS] Displaying cluster weight threshold");
  logger_->report("{}", GetVectorString(thr_cluster_weight));

  // create the coarsener cluster
  auto tritonpart_coarsener
      = std::make_shared<Coarsener>(num_parts_,
                                    thr_coarsen_hyperedge_size_skip_,
                                    thr_coarsen_vertices_,
                                    thr_coarsen_hyperedges_,
                                    coarsening_ratio_,
                                    max_coarsen_iters_,
                                    adj_diff_ratio_,
                                    thr_cluster_weight,
                                    seed_,
                                    coarsen_order_,
                                    tritonpart_evaluator,
                                    logger_);

  tritonpart_coarsener->SetCommunityAttrOpt(community_attr_);

  // create the initial partitioning class
  auto tritonpart_partitioner = std::make_shared<Partitioner>(
      num_parts_, seed_, tritonpart_evaluator, logger_);

  // create the refinement classes
  // We have four types of refiner
  // (1) greedy refinement. try to one entire hyperedge each time
  auto greedy_refiner = std::make_shared<GreedyRefine>(num_parts_,
                                                       refiner_iters_,
                                                       path_timing_factor_,
                                                       path_snaking_factor_,
                                                       max_moves_,
                                                       tritonpart_evaluator,
                                                       logger_);

  // (2) ILP-based partitioning (only for two-way since k-way ILP partitioning
  // is too timing-consuming)
  auto ilp_refiner = std::make_shared<IlpRefine>(num_parts_,
                                                 refiner_iters_,
                                                 path_timing_factor_,
                                                 path_snaking_factor_,
                                                 max_moves_,
                                                 tritonpart_evaluator,
                                                 logger_);

  // (3) direct k-way FM
  auto k_way_fm_refiner = std::make_shared<KWayFMRefine>(num_parts_,
                                                         refiner_iters_,
                                                         path_timing_factor_,
                                                         path_snaking_factor_,
                                                         max_moves_,
                                                         total_corking_passes_,
                                                         tritonpart_evaluator,
                                                         logger_);

  // (4) k-way pair-wise FM
  auto k_way_pm_refiner = std::make_shared<KWayPMRefine>(num_parts_,
                                                         refiner_iters_,
                                                         path_timing_factor_,
                                                         path_snaking_factor_,
                                                         max_moves_,
                                                         total_corking_passes_,
                                                         tritonpart_evaluator,
                                                         logger_);

  // create the multi-level class
  auto tritonpart_mlevel_partitioner
      = std::make_shared<MultilevelPartitioner>(num_parts_,
                                                v_cycle_flag_,
                                                num_initial_solutions_,
                                                num_best_initial_solutions_,
                                                num_vertices_threshold_ilp_,
                                                max_num_vcycle_,
                                                num_coarsen_solutions_,
                                                seed_,
                                                tritonpart_coarsener,
                                                tritonpart_partitioner,
                                                k_way_fm_refiner,
                                                k_way_pm_refiner,
                                                greedy_refiner,
                                                ilp_refiner,
                                                tritonpart_evaluator,
                                                logger_);

  if (timing_aware_flag_ == true) {
    // Initialize the timing on original_hypergraph_
    tritonpart_evaluator->InitializeTiming(original_hypergraph_);
  }
  // Use coarsening to do preprocessing step
  // Build the hypergraph used to call multi-level partitioner
  // (1) remove single-vertex hyperedge
  // (2) remove lager hyperedge
  // (3) detect parallel hyperedges
  // (4) handle group information
  // (5) group fixed vertices based on each block
  // group vertices based on group_attr_
  // the original_hypergraph_ will be modified by the Group Vertices command
  // We will store the mapping relationship of vertices between
  // original_hypergraph_ and hypergraph_
  tritonpart_coarsener->SetThrCoarsenHyperedgeSizeSkip(global_net_threshold_);
  hypergraph_
      = tritonpart_coarsener->GroupVertices(original_hypergraph_, group_attr_);
  tritonpart_coarsener->SetThrCoarsenHyperedgeSizeSkip(
      thr_coarsen_hyperedge_size_skip_);

  hypergraph_->SetVertexNames(vertex_names_);

  // partition on the processed hypergraph
  std::vector<int> solution = tritonpart_mlevel_partitioner->Partition(
      hypergraph_, upper_block_balance, lower_block_balance);

  // Translate the solution of hypergraph to original_hypergraph_
  // solution to solution_
  solution_.clear();
  solution_.resize(original_hypergraph_->GetNumVertices());
  std::fill(solution_.begin(), solution_.end(), -1);
  for (int cluster_id = 0; cluster_id < hypergraph_->GetNumVertices();
       cluster_id++) {
    const int part_id = solution[cluster_id];
    for (const auto& v : hypergraph_->GetVertexCAttr(cluster_id)) {
      solution_[v] = part_id;
    }
  }

  // Perform the last-minute refinement
  tritonpart_coarsener->SetThrCoarsenHyperedgeSizeSkip(global_net_threshold_);
  tritonpart_mlevel_partitioner->VcycleRefinement(
      hypergraph_, upper_block_balance, lower_block_balance, solution_);

  // evaluate on the original hypergraph
  // tritonpart_evaluator->CutEvaluator(original_hypergraph_, solution_,
  // true);
  tritonpart_evaluator->ConstraintAndCutEvaluator(original_hypergraph_,
                                                  solution_,
                                                  ub_factor_,
                                                  base_balance_,
                                                  group_attr_,
                                                  true);

  // generate the timing report
  if (timing_aware_flag_ == true) {
    logger_->report("[STATUS] Displaying timing path cuts statistics");
    PathStats path_stats
        = tritonpart_evaluator->GetTimingCuts(original_hypergraph_, solution_);
    tritonpart_evaluator->PrintPathStats(path_stats);
  }

  // print the runtime
  auto end_timestamp_global = std::chrono::high_resolution_clock::now();
  double total_global_time
      = std::chrono::duration_cast<std::chrono::nanoseconds>(
            end_timestamp_global - start_time_stamp_global)
            .count();
  total_global_time *= 1e-9;
  logger_->info(PAR,
                109,
                "The runtime of multi-level partitioner : {} seconds",
                total_global_time);
}

}  // namespace par

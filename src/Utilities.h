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
// This file includes the basic utility functions for operations
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <map>
#include <string>
#include <vector>
#include <iostream>

#ifdef LOAD_CPLEX
// for ILP solver in CPLEX
#include "ilcplex/cplex.h"
#include "ilcplex/ilocplex.h"
#endif

#include "odb/db.h"
#include "utl/Logger.h"

namespace par {

// Matrix is a two-dimensional vectors
template <typename T>
using Matrix = std::vector<std::vector<T>>;

struct Rect
{
  // all the values are in db unit
  int lx = 0;
  int ly = 0;
  int ux = 0;
  int uy = 0;

  Rect(int lx_, int ly_, int ux_, int uy_) : lx(lx_), ly(ly_), ux(ux_), uy(uy_)
  {
  }

  // check if the Rect is valid
  bool IsValid() const { return ux > lx && uy > ly; }

  // reset the fence
  void Reset()
  {
    lx = 0;
    ly = 0;
    ux = 0;
    uy = 0;
  }
};

// Define the type for vertices
enum VertexType
{
  COMB_STD_CELL,  // combinational standard cell
  SEQ_STD_CELL,   // sequential standard cell
  MACRO,          // hard macros
  PORT            // IO ports
};

std::string GetVectorString(const std::vector<float>& vec);

// Split a string based on deliminator : empty space and ","
std::vector<std::string> SplitLine(const std::string& line);

// Add right vector to left vector
void Accumulate(std::vector<float>& a, const std::vector<float>& b);

// weighted sum
std::vector<float> WeightedSum(const std::vector<float>& a,
                               float a_factor,
                               const std::vector<float>& b,
                               float b_factor);

// divide the vector
std::vector<float> DivideFactor(const std::vector<float>& a, float factor);

// divide the vectors element by element
std::vector<float> DivideVectorElebyEle(const std::vector<float>& emb,
                                        const std::vector<float>& factor);

// multiplty the vector
std::vector<float> MultiplyFactor(const std::vector<float>& a, float factor);

// operation for two vectors +, -, *,  ==, <
std::vector<float> operator+(const std::vector<float>& a,
                             const std::vector<float>& b);

std::vector<float> operator*(const std::vector<float>& a, float factor);

std::vector<float> operator-(const std::vector<float>& a,
                             const std::vector<float>& b);

std::vector<float> operator*(const std::vector<float>& a,
                             const std::vector<float>& b);

bool operator<(const std::vector<float>& a, const std::vector<float>& b);

bool operator<=(const Matrix<float>& a, const Matrix<float>& b);

bool operator==(const std::vector<float>& a, const std::vector<float>& b);

// Basic functions for a vector
std::vector<float> abs(const std::vector<float>& a);

float norm2(const std::vector<float>& a);

float norm2(const std::vector<float>& a, const std::vector<float>& factor);

// ILP-based Partitioning Instance
// Call ILP Solver to partition the design
bool ILPPartitionInst(
    int num_parts,
    int vertex_weight_dimension,
    std::vector<int>& solution,
    const std::map<int, int>& fixed_vertices,     // vertex_id, block_id
    const Matrix<int>& hyperedges,                // hyperedges
    const std::vector<float>& hyperedge_weights,  // one-dimensional
    const Matrix<float>& vertex_weights,          // two-dimensional
    const Matrix<float>& upper_block_balance,
    const Matrix<float>& lower_block_balance);

// Call CPLEX to solve the ILP Based Partitioning
#ifdef LOAD_CPLEX
bool OptimalPartCplex(
    int num_parts,
    int vertex_weight_dimension,
    std::vector<int>& solution,
    const std::map<int, int>& fixed_vertices,     // vertex_id, block_id
    const Matrix<int>& hyperedges,                // hyperedges
    const std::vector<float>& hyperedge_weights,  // one-dimensional
    const Matrix<float>& vertex_weights,          // two-dimensional
    const Matrix<float>& upper_block_balance,
    const Matrix<float>& lower_block_balance);
#endif

// Define data structure for logical hierarchy
class HierarchyNode;
using HierNodePtr = std::shared_ptr<HierarchyNode>;

class HierarchyNode
{
 public:
  HierarchyNode() = default;
  HierarchyNode(const std::string& name) : name_(name) {}
  HierarchyNode(const std::string& name,
                std::vector<odb::dbModInst*> children_mods,
                std::vector<odb::dbInst*> children_insts)
      : name_(name),
        children_mods_(children_mods),
        children_insts_(children_insts)
  {
    parent_ = nullptr;  // Set the parent to be nullptr for the root node
    active_ = true;     // Set the node to be active by default
  }

  void AddChild(std::shared_ptr<HierarchyNode> child)
  {
    children_.push_back(child);
    child->parent_ = std::make_shared<HierarchyNode>(*this);
  }

  bool IsLeaf() const { return children_.empty(); }

  void AddParent(std::shared_ptr<HierarchyNode> parent)
  {
    this->parent_ = parent;
    parent->children_.push_back(std::make_shared<HierarchyNode>(*this));
  }

  HierNodePtr GetParent() const { return parent_; }

  std::vector<odb::dbModInst*> GetChildrenMods() const
  {
    return std::move(children_mods_);
  }

  std::vector<HierNodePtr> GetChildrenNodes() const
  {
    return std::move(children_);
  }

  std::vector<odb::dbInst*> GetChildrenInsts() const
  {
    return std::move(children_insts_);
  }

  void AddChildrenInsts(std::vector<odb::dbInst*> children_insts)
  {
    children_insts_.insert(
        children_insts_.end(), children_insts.begin(), children_insts.end());
  }

  void RemoveChildrenInsts()
  {
    children_insts_.clear();
  }

  void AddChildrenInst(odb::dbInst* child_inst)
  {
    children_insts_.emplace_back(child_inst);
  }

  bool GetActive() const { return active_; }

  void SetActive(bool active) { active_ = active; }

  void SetVertexId(int vertex_id) { vertex_id_ = vertex_id; }

  int GetVertexId() const { return vertex_id_; }

  std::string GetName() const { return name_; }

  int GetTotalInstances() const { return total_instances_; }

 private:
  bool active_ = true;                          // If the node is active
  std::string name_;                            // Save the name of the node
  std::vector<HierNodePtr> children_;           // Save the children
  std::shared_ptr<HierarchyNode> parent_;       //  Save the parent node
  std::vector<odb::dbModInst*> children_mods_;  // Save the children modules
  std::vector<odb::dbInst*> children_insts_;    // Save the children instances
  int total_instances_ = 0;  // Save how many instances in this node
  int vertex_id_ = -1;       // Save the vertex id in the tree

  friend class HierarchyTree;
};

class HierarchyTree;
using HierTreePtr = std::shared_ptr<HierarchyTree>;

class HierarchyTree
{
 public:
  HierarchyTree() = default;
  HierarchyTree(std::shared_ptr<HierarchyNode> root) : root_(root)
  {
    nodes_.push_back(root);
    name2node_[root->name_] = root;
    num_vertices_++;
  }
  HierarchyTree(const HierarchyTree& other) = default;

  void SetRoot(HierNodePtr root)
  {
    root_ = root;
  }

  // Add a node to the tree
  void AddNode(std::shared_ptr<HierarchyNode> node)
  {
    nodes_.push_back(node);
    name2node_[node->name_] = node;
    num_vertices_++;
  }
  // Find leaf nodes
  std::vector<HierNodePtr> GetLeafNodes() const
  {
    std::vector<HierNodePtr> leaf_nodes;
    for (auto& node : nodes_) {
      if (node->GetActive() && node->children_.empty()) {
        leaf_nodes.push_back(node);
      }
    }
    return leaf_nodes;
  }

  int GetNumVertices() const { return num_vertices_; }

  HierNodePtr GetNode(int idx) const { return nodes_[idx]; }

  HierNodePtr GetNodeFromName(const std::string& name) const
  {
    return name2node_.at(name);
  }

  std::vector<HierNodePtr> GetNodes() const
  {
    std::vector<HierNodePtr> nodes;
    for (int i = 0; i < nodes_.size(); i++) {
      if (nodes_[i]->GetActive()) {
        nodes.push_back(nodes_[i]);
      }
    }
    return nodes;
  }

  int GetNumVerticesWithInstances() const
  {
    int num_vertices = 0;
    for (auto& node : nodes_) {
      if (node->GetActive() && node->GetChildrenInsts().size() > 0) {
        num_vertices++;
      }
    }
    return num_vertices;
  }

  int GetTotalInstancesInTree() const 
  {
    int total_instances = 0;
    for (auto& node : nodes_) {
      if (node->GetActive()) {
        total_instances += node->GetChildrenInsts().size();
      }
    }
    return total_instances;
  }

  void DeleteLeafNode(HierNodePtr node)
  {
    // Delete the node from the tree
    auto parent = node->parent_;
    auto it
        = std::find(parent->children_.begin(), parent->children_.end(), node);
    parent->children_.erase(it);
    // Reduce the number of vertices
    num_vertices_--;
    // Deactive the node
    node->SetActive(false);
  }

  // Find parent nodes of all leaf nodes
  std::vector<std::shared_ptr<HierarchyNode>> GetParentNodesOfLeafNodes() const
  {
    std::map<std::string, int> vertex_idx_map;
    std::vector<std::shared_ptr<HierarchyNode>> parent_nodes;
    std::vector<std::shared_ptr<HierarchyNode>> leaf_nodes = GetLeafNodes();

    for (auto& node : leaf_nodes) {
      // add the parent node to the vector if hash map does not contain it
      std::string parent_name = node->parent_->GetName();
      if (vertex_idx_map.find(parent_name) == vertex_idx_map.end()) {
        vertex_idx_map[parent_name] = 1;
        parent_nodes.push_back(GetNodeFromName(parent_name));
      }
    }

    return parent_nodes;
  }

  void GenerateNodeIdMap()
  {
    node_name2id_.clear();
    for (int i = 0; i < nodes_.size(); i++) {
      if (nodes_[i]->GetActive()) {
        node_name2id_[nodes_[i]->name_] = i;
      }
    }
  }

  // Pretty print the tree structure starting from root
  void PrintTree(HierNodePtr node, int indent = 0, char symbol = ' ')
  {
    if (node == nullptr) {
      return;
    }

    constexpr int indentSize = 4;
    // Print the current node
    logger_->report("{}", std::string(indent, symbol) + node->GetName());
    for (auto& child : node->children_) {
      if (child->GetActive()) {
        PrintTree(child, indent + indentSize, symbol);
      }
    }
  }

  // Delete tree
  void DeleteTree(HierNodePtr node)
  {
    if (node == nullptr) {
      return;
    }

    for (auto& child : node->children_) {
      DeleteTree(child);
    }
    node->children_.clear();
  }

  HierNodePtr GetRoot() const { return root_; }

  // Given a node in the tree find the cluster from the node
  std::vector<HierNodePtr> GetClusterAroundNode(HierNodePtr node);
 private:
  int num_vertices_ = 0;
  std::shared_ptr<HierarchyNode> root_;
  std::unordered_map<std::string, std::shared_ptr<HierarchyNode>> name2node_;
  std::vector<std::shared_ptr<HierarchyNode>> nodes_;
  std::unordered_map<std::string, int> node_name2id_;
  utl::Logger* logger_;
};

class HierDendogramNode;
using HierDendoNodePtr = std::shared_ptr<HierDendogramNode>;

class HierDendogramNode
{
 public:
  HierDendogramNode() = default;
  HierDendogramNode(const std::vector<int>& vertex_ids, float rent_param, int level, int node_id)
      : vertex_ids_(vertex_ids), rent_param_(rent_param), level_(level), node_id_(node_id)
  {
  }

  HierDendogramNode(const std::vector<HierDendoNodePtr>& children,
                    const std::vector<int>& vertex_ids,
                    float rent_param,
                    int level)
      : children_(children),
        vertex_ids_(vertex_ids),
        rent_param_(rent_param),
        level_(level)
  {
  }

  int GetNodeId() const
  {
    return node_id_;
  }

  void SetNodeId(int node_id)
  {
    node_id_ = node_id;
  }

  void SetLevel(int level)
  {
    level_ = level;
  }

  int GetLevel() const
  {
    return level_;
  }

  void AddChild(HierDendoNodePtr child)
  {
    children_.push_back(child);
  }

  void AddVertexId(int vertex_id)
  {
    vertex_ids_.push_back(vertex_id);
  }

  void SetRentParam(float rent_param)
  {
    rent_param_ = rent_param;
  }

  float GetRentParam() const
  {
    return rent_param_;
  }

  std::vector<HierDendoNodePtr> GetChildren() const
  {
    return children_;
  }

  std::vector<int> GetVertexIds() const
  {
    return vertex_ids_;
  }

  int GetTotalVertices() const
  {
    return vertex_ids_.size();
  }

  void SetVertexIds(std::vector<int> vertex_ids)
  {
    vertex_ids_ = vertex_ids;
  }

 private: 
  std::vector<HierDendoNodePtr> children_;
  std::vector<int> vertex_ids_;
  float rent_param_;
  int level_; 
  int node_id_;
};

class HierDendogram;
using HierDendoPtr = std::shared_ptr<HierDendogram>;

class HierDendogram
{
 public:
  HierDendogram() = default;
  HierDendogram(int max_levels) : max_levels_(max_levels) {}
  HierDendogram(HierDendoNodePtr root) : root_(root)
  {
    nodes_.push_back(root);
    num_nodes_++;
  }

  HierDendogram(const HierDendogram& other) = default;

  HierDendogram(HierDendoNodePtr root, std::vector<HierDendoNodePtr> nodes)
      : root_(root), nodes_(nodes)
  {
    num_nodes_ = nodes.size();
  }

  HierDendogram(HierDendoNodePtr root,
                std::vector<HierDendoNodePtr> nodes,
                utl::Logger* logger)
      : root_(root), nodes_(nodes)
  {
    num_nodes_ = nodes.size();
    logger_ = logger;
  }

  void SetMaxLevels(int max_levels)
  {
    max_levels_ = max_levels;
  }

  int GetMaxLevels() const
  {
    return max_levels_;
  }

  void SetRoot(HierDendoNodePtr root)
  {
    root_ = root;
  }

  // Add a node to the tree
  void AddNode(HierDendoNodePtr node)
  {
    nodes_.push_back(node);
    num_nodes_++;
  }

  int GetNumNodes() const { return num_nodes_; }

  void SetNodes(std::vector<HierDendoNodePtr> nodes)
  {
    nodes_ = nodes;
  }

  void SetNodeRentParam(int idx, float rent_param)
  {
    nodes_[idx]->SetRentParam(rent_param);
  }

  float GetNodeRentParam(int idx) const
  {
    return nodes_[idx]->GetRentParam();
  }

  HierDendoNodePtr GetNode(int idx) const { return nodes_[idx]; }

  void ClusterAtLevel(int level);

  float GetAverageRentParam(int level) const;

  std::vector<HierDendoNodePtr> GetNodesAtLevel(int level) const 
  {
    std::vector<HierDendoNodePtr> nodes_at_level;
    for (int i = 0; i < nodes_.size(); i++) {
      if (nodes_[i]->GetLevel() == level) {
        nodes_at_level.push_back(nodes_[i]);
      }
    }
    return nodes_at_level;
  }

  float GetAvgRentAtLevel(int level) const 
  {
    float avg_rent = 0.0;
    int num_nodes = 0;
    for (int i = 0; i < nodes_.size(); i++) {
      if (nodes_[i]->GetLevel() == level) {
        avg_rent += nodes_[i]->GetRentParam() * nodes_[i]->GetTotalVertices();
        num_nodes += nodes_[i]->GetTotalVertices();
      }
    }
    return avg_rent / num_nodes;
  }

 private: 
  HierDendoNodePtr root_;
  std::vector<HierDendoNodePtr> nodes_;
  utl::Logger* logger_;
  int num_nodes_ = 0;
  int max_levels_ = 0;
};

}  // namespace par

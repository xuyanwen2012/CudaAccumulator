#include "bh_tree.h"

#include <iostream>
#include <algorithm>
#include <queue>
#include <vector_types.h>

using namespace barnes_hut;

struct accumulator_handle
{
	float* result_addr;
	float x;
	float y;
};

void tree_node::insert_body(const body_ptr& body_ptr)
{
	if (is_leaf_)
	{
		if (is_empty())
		{
			content = body_ptr;
			return;
		}

		// more than 1 particles are allocated into this node, need to split,
		// then re-insert the current content to the deeper levels
		split();

		const auto quadrant = static_cast<size_t>(determine_quadrant(content->pos()));
		children.at(quadrant)->insert_body(content);

		content.reset();
	}

	const auto new_quadrant = static_cast<size_t>(determine_quadrant(body_ptr->pos()));
	children.at(new_quadrant)->insert_body(body_ptr);
}

tree_node::direction tree_node::determine_quadrant(const vec2& pos) const
{
	const auto cx = bounding_box.center.real();
	const auto cy = bounding_box.center.imag();
	const auto x = pos.real();
	const auto y = pos.imag();

	if (x < cx)
	{
		if (y < cy)
		{
			return direction::sw;
		}
		return direction::nw;
	}
	if (y < cy)
	{
		return direction::se;
	}
	return direction::ne;
}

void tree_node::split()
{
	is_leaf_ = false;

	const auto hw = bounding_box.size.real() / 2.0f;
	const auto hh = bounding_box.size.imag() / 2.0f;
	const auto cx = bounding_box.center.real();
	const auto cy = bounding_box.center.imag();

	const auto next_level = level + 1;
	quadtree::depth = std::max(quadtree::depth, level);
	quadtree::num_nodes += 4;

	const auto my_uid = uid * 10;

	const auto sw = new tree_node{my_uid + 0, rect<float>{cx - hw / 2.0f, cy - hh / 2.0f, hw, hh}, next_level};
	const auto se = new tree_node{my_uid + 1, rect<float>{cx + hw / 2.0f, cy - hh / 2.0f, hw, hh}, next_level};
	const auto nw = new tree_node{my_uid + 2, rect<float>{cx - hw / 2.0f, cy + hh / 2.0f, hw, hh}, next_level};
	const auto ne = new tree_node{my_uid + 3, rect<float>{cx + hw / 2.0f, cy + hh / 2.0f, hw, hh}, next_level};

	children[0] = sw;
	children[1] = se;
	children[2] = nw;
	children[3] = ne;
}

quadtree::quadtree() :
	num_particles(0),
	root_(tree_node{1, rect<float>{0.5f, 0.5f, 1.0f, 1.0f}, 0})
{
}

void quadtree::allocate_node_for_particle(const body_ptr& body_ptr)
{
	++num_particles;
	root_.insert_body(body_ptr);
}

void quadtree::compute_center_of_mass()
{
	std::queue<tree_node*> queue;
	std::vector<tree_node*> list;

	queue.push(&root_);
	while (!queue.empty())
	{
		const auto cur = queue.front();
		queue.pop();

		if (!cur->is_leaf_)
		{
			for (auto child : cur->children)
			{
				queue.push(child);
			}
		}

		list.push_back(cur);
	}

	std::for_each(list.rbegin(), list.rend(),
	              [&](tree_node* node)
	              {
		              // sum the masses
		              float mass_sum = 0.0f;
		              std::complex<float> weighted_pos_sum{0.0f, 0.0f};
		              if (node->is_leaf_)
		              {
			              if (node->content != nullptr)
			              {
				              mass_sum = node->content->mass;
				              weighted_pos_sum = node->content->pos() * node->content->mass;
			              }
		              }
		              else
		              {
			              for (const tree_node* child : node->children)
			              {
				              mass_sum += child->node_mass;
				              weighted_pos_sum += child->weighted_pos;
			              }
		              }

		              node->node_mass = mass_sum;
		              node->weighted_pos = weighted_pos_sum;
	              });
}

void inner_dfs_accumulate(const tree_node* current, accumulator_handle* acc, const float theta)
{
	const vec2 pos = {acc->x, acc->y};

	//printf("%d: \n", current->uid);

	if (current->is_leaf())
	{
		if (current->is_empty())
		{
			return;
		}

		//printf("  += (%f,%f)\n", current->content->x, current->content->y);

		accumulator_accumulate(current->content->x,
		                       current->content->y,
		                       current->content->mass,
		                       acc);
	}
	else if (quadtree::check_theta(current, pos, theta))
	{
		const auto cm = current->center_of_mass();
		//printf("  += (%f,%f) (Node)\n", cm.real(), cm.imag());
		accumulator_accumulate(cm.real(),
		                       cm.imag(),
		                       current->node_mass,
		                       acc);
	}
	else
	{
		for (const tree_node* child : current->children)
		{
			inner_dfs_accumulate(child, acc, theta);
		}
	}
}

void quadtree::compute_force_accumulator(accumulator_handle* acc,
                                         const float theta) const
{
	inner_dfs_accumulate(&root_, acc, theta);
}

bool quadtree::check_theta(const tree_node* node, const vec2& pos, const float theta)
{
	const std::complex<float> com = node->center_of_mass();

	const std::complex<float> distance = com - pos;
	const auto norm = abs(distance);

	const auto geo_size = node->bounding_box.size.real();
	return geo_size / norm < theta;
}


size_t quadtree::depth = 0;

size_t quadtree::num_nodes = 1;

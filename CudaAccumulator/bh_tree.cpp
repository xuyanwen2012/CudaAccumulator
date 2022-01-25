#include "bh_tree.h"

#include <iostream>
#include <algorithm>
#include <queue>
#include <vector_types.h>


struct accumulator_handle
{
	float x;
	float y;
};

void barnes_hut::tree_node::insert_body(const body_ptr& body_ptr)
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

barnes_hut::tree_node::direction barnes_hut::tree_node::determine_quadrant(const vec2& pos) const
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

void barnes_hut::tree_node::split()
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

barnes_hut::quadtree::quadtree() :
	num_particles(0),
	root_(tree_node{1, rect<float>{0.5f, 0.5f, 1.0f, 1.0f}, 0})
{
}

void barnes_hut::quadtree::allocate_node_for_particle(const body_ptr& body_ptr)
{
	++num_particles;
	root_.insert_body(body_ptr);
}

void barnes_hut::quadtree::compute_center_of_mass()
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
		              std::complex<float> weighted_pos_sum{0, 0};
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

void inner_dfs_accumulate(barnes_hut::tree_node* current, accumulator_handle* acc, const float theta)
{
	using namespace barnes_hut;

	const vec2 pos = {acc->x, acc->y};

	if (current->is_leaf())
	{
		if (current->is_empty())
		{
			return;
		}

		accumulator_accumulate(current->content->x,
		                       current->content->y,
		                       current->content->mass,
		                       acc);
	}
	else if (quadtree::check_theta(current, pos, theta))
	{
		const auto cm = current->center_of_mass();
		accumulator_accumulate(cm.real(),
		                       cm.imag(),
		                       current->node_mass,
		                       acc);
	}
	else
	{
		for (tree_node* child : current->children)
		{
			// printf("%d\n", child->uid);
			inner_dfs_accumulate(child, acc, theta);
		}
	}
}

std::complex<float> barnes_hut::quadtree::compute_force_accumulator(accumulator_handle* acc,
                                                                    const vec2& pos,
                                                                    const float theta)
{
	float2 us{};

	accumulator_set_constants_and_result_address(pos.real(), pos.imag(), &us.x, acc);

	inner_dfs_accumulate(&root_, acc, theta);

	return {us.x, us.y};
}

bool barnes_hut::quadtree::check_theta(const tree_node* node, const vec2& pos, const float theta)
{
	const std::complex<float> com = node->center_of_mass();

	const std::complex<float> distance = com - pos;
	const auto norm = abs(distance);

	const auto geo_size = node->bounding_box.size.real();
	return geo_size / norm < theta;
}


size_t barnes_hut::quadtree::depth = 0;

size_t barnes_hut::quadtree::num_nodes = 1;

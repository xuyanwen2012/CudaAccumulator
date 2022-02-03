#include "kd_tree.h"

#include <algorithm>

#include "bh_tree.h"

node::node(const float x, const float y) : point{x, y}, left(), right()
{
}

kd_tree::kd_tree() : root_()
{
}

node* inner_insert_point(node* current, const point_t& point, const unsigned depth)
{
	if (current == nullptr)
	{
		return new node(point[0], point[1]);
	}

	// cd == 0 is x-axis, 1 is y-axis.
	const unsigned cd = depth % k;
	if (point[cd] < current->point[cd])
	{
		current->left = inner_insert_point(current->left, point, depth + 1);
	}
	else
	{
		current->right = inner_insert_point(current->right, point, depth + 1);
	}

	return current;
}

void kd_tree::insert_point(const point_t& point)
{
	root_ = inner_insert_point(root_, point, 0);
}

float inner_find_min(const node* current, const unsigned dim, const unsigned depth)
{
	const unsigned cd = depth % k;

	if (cd == dim)
	{
		// minimum can’t be in the right subtree
		if (current->left == nullptr)
		{
			return current->point[cd];
		}

		return inner_find_min(current->left, dim, depth + 1);
	}

	// minimum could be in either subtree
	float val_left = std::numeric_limits<float>::max();
	float val_right = std::numeric_limits<float>::max();
	if (current->left != nullptr)
	{
		val_left = inner_find_min(current->left, dim, depth + 1);
	}
	if (current->right != nullptr)
	{
		val_right = inner_find_min(current->right, dim, depth + 1);
	}

	return std::min(val_left, val_right);
}

float kd_tree::find_min(const unsigned dim) const
{
	return inner_find_min(root_, dim, 0);
}

float distance(const point_t& q, const point_t& p)
{
	const std::complex<float> distance (q[0] - p[0], q[1] - p[1]);
	return abs(distance);
}

point_t inner_nearest_neighbor(node* current, const point_t& p, const unsigned depth, barnes_hut::rect<float> rect)
{
	// if this bounding box is too far, do nothing
}

point_t kd_tree::find_nearest_neighbor(const point_t& point)
{
	return inner_nearest_neighbor(root_, point, 0, {0.5f, 0.5f, 1.0f, 1.0f});
}

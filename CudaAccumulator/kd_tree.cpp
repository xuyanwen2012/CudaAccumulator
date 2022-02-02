#include "kd_tree.h"

node::node(const float x, const float y) : point{x, y}, left(), right()
{
}

kd_tree::kd_tree() : root_()
{
}

node* kd_tree::insert_point(const point_t& point)
{
	return inner_insert_point(root_, point, 0);
}

node* kd_tree::inner_insert_point(node* current, const point_t& point, const unsigned depth)
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

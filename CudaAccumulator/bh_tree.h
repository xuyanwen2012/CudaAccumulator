#pragma once

#include <array>
#include <complex>
#include <memory>
#include <type_traits>

#include "body.h"
#include "accumulator.h"

// TODO macros about runtime info

namespace barnes_hut
{
	template <typename T,
	          typename = std::enable_if<std::is_floating_point_v<T>, T>>
	struct rect
	{
		rect() = default;

		rect(const T cx, const T cy, const T w, const T h) : center(cx, cy), size(w, h)
		{
		}

		std::complex<T> center;
		std::complex<T> size;
	};

	using vec2 = std::complex<float>;
	using body_ptr = std::shared_ptr<body<float>>;

	struct tree_node
	{
		enum class direction { sw = 0, se, nw, ne };

		friend class quadtree;

		tree_node() = delete;

		tree_node(const int uid, const rect<float> bound, const size_t level)
			: uid(uid), level(level), bounding_box(bound), node_mass(0), is_leaf_(true)
		{
		}

		int uid;
		size_t level;

		/// <summary>
		/// I used center point as the position.
		///	Also the entire boarder of the whole universe is between [0..1]
		/// </summary>
		rect<float> bounding_box;

		/// <summary>
		///
		/// </summary>
		body_ptr content;

		/// <summary>
		///	 2 | 3
		/// ---+---
		///	 0 | 1
		/// </summary>
		std::array<tree_node*, 4> children{};

		/// <summary>
		/// This field stores the total mass of this node and its descendants
		/// </summary>
		float node_mass;

		/// <summary>
		///	The total sum of this node's and its descendants 'Position * mass'
		/// This is used to compute the center of mass, use it divide by 'node_mass'
		/// </summary>
		std::complex<float> weighted_pos;

		/// <summary>
		///
		/// </summary>
		bool is_empty() const { return content == nullptr; }

		/// <summary>
		///
		/// </summary>
		std::complex<float> center_of_mass() const { return weighted_pos / node_mass; }

		bool is_leaf() const { return is_leaf_; }

	private:
		bool is_leaf_;
		void insert_body(const body_ptr& body_ptr);
		direction determine_quadrant(const vec2& pos) const;
		void split();
	};


	class quadtree
	{
	public:
		/// <summary>
		/// create a empty quadtree with only a node.
		/// </summary>
		quadtree();

		/// <summary>
		/// Use this function to insert a body into the quadtree.
		/// </summary>
		/// <param name="body_ptr"></param>
		void allocate_node_for_particle(const body_ptr& body_ptr);

		/// <summary>
		/// Once every particles are allocated into the quadtree, we can
		///	compute the center of masses and the quadtree is ready for
		///	inquiry.
		/// </summary>
		void compute_center_of_mass();

		std::complex<float> compute_force_accumulator(accumulator_handle* acc, const vec2& pos, float theta) const;

		// some statistical things
		size_t num_particles;
		static size_t num_nodes;
		static size_t depth;

		static inline bool check_theta(const tree_node* node, const vec2& pos, float theta);

	private:
		tree_node root_;
	};
}

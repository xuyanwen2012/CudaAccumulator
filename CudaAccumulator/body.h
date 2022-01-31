#pragma once

#include <type_traits>
#include <complex>

template <typename T,
          typename = std::enable_if<std::is_floating_point_v<T>, T>>
struct body
{
	body(T x, T y, T mass) : x(x), y(y), mass(mass)
	{
	}

	T x;
	T y;
	T mass;

	// Only needed for the current BH tree implementation,
	// in future it will be replaced
	std::complex<T> pos() { return std::complex<T>(x, y); }
};

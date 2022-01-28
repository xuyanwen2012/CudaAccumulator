#pragma once

#include <type_traits>
#include <complex>

template <typename T,
          typename = std::enable_if<std::is_floating_point<T>::value, T>>
struct body
{
	body(T x, T y, T mass) : x(x), y(y), mass(mass)
	{
	}

	T x;
	T y;
	T mass;

	std::complex<T> pos() { return std::complex<T>(x, y); }
};

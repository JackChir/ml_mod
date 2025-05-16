#ifndef _WEIGHT_INITILIZER_HPP_
#define _WEIGHT_INITILIZER_HPP_
#include <random>

#include "mat.hpp"

static std::default_random_engine ge;

template<typename target_t, typename rand_distrib_t>
struct do_init
{
	static void cal(target_t& mt_or_val, rand_distrib_t& ud)
	{
		mt_or_val = ud(ge);
	}
};

template<int row_num, int col_num, typename val_t, typename rand_distrib_t>
struct do_init<mat<row_num, col_num, val_t>, rand_distrib_t >
{
	static void cal(mat<row_num, col_num, val_t>& mt, rand_distrib_t& ud)
	{
		for (int i = 0; i < row_num; ++i)
		{
			for (int j = 0; j < col_num; ++j)
			{
				do_init<val_t, rand_distrib_t>::cal(mt.get(i, j), ud);
			}
		}
	}
};

template<typename init_name_t>
struct weight_initilizer 
{
	template<int row_num, int col_num, typename val_t>
	static void cal(mat<row_num, col_num, val_t>& mt, const double& d1 = 0., const double& d2 = 1.)
	{
		static std::uniform_real_distribution <double> ud(d1, d2);
		for (int i = 0; i < row_num; ++i)
		{
			for (int j = 0; j < col_num; ++j)
			{
				do_init<val_t, std::uniform_real_distribution<double> >::cal(mt.get(i, j), ud);
			}
		}
	}
};

template<>
struct weight_initilizer<class XavierGaussian>
{
	template<int row_num, int col_num, typename val_t>
	static void cal(mat<row_num, col_num, val_t>& mt) 
	{
		static std::normal_distribution <double> ud(0., sqrtl(2. / (row_num + col_num)));

		for (int i = 0; i < row_num; ++i)
		{
			for (int j = 0; j < col_num; ++j)
			{
				do_init<val_t, std::normal_distribution<double> >::cal(mt.get(i, j), ud);
			}
		}
	}
};

template<>
struct weight_initilizer<class XavierMean>
{
	template<int row_num, int col_num, typename val_t>
	static void cal(mat<row_num, col_num, val_t>& mt)
	{
		double r = sqrtl(6. / (row_num + col_num));
		static std::uniform_real_distribution<double> ud(-r, r);
		for (int i = 0; i < row_num; ++i)
		{
			for (int j = 0; j < col_num; ++j)
			{
				do_init<val_t, std::uniform_real_distribution<double> >::cal(mt.get(i, j), ud);
			}
		}
	}
};

template<>
struct weight_initilizer<class HeGaussian>
{
	template<int row_num, int col_num, typename val_t>
	static void cal(mat<row_num, col_num, val_t>& mt)
	{
		static std::default_random_engine e;
		static std::normal_distribution <double> ud(0., sqrtl(2. / col_num));
		for (int i = 0; i < row_num; ++i)
		{
			for (int j = 0; j < col_num; ++j)
			{
				do_init<val_t, std::normal_distribution<double> >::cal(mt.get(i, j), ud);
			}
		}
	}
};

template<>
struct weight_initilizer<class HeMean>
{
	template<int row_num, int col_num, typename val_t>
	static void cal(mat<row_num, col_num, val_t>& mt)
	{
		double r = sqrtl(6. / col_num);
		static std::uniform_real_distribution<double> ud(-r, r);
		for (int i = 0; i < row_num; ++i)
		{
			for (int j = 0; j < col_num; ++j)
			{
				do_init<val_t, std::uniform_real_distribution<double> >::cal(mt.get(i, j), ud);
			}
		}
	}
};

#endif

#ifndef _UPDATE_METHODS_HPP_
#define _UPDATE_METHODS_HPP_

#include <math.h>

#include "mat.hpp"


template<typename target_t>
struct gd
{
	typename target_t::type lr;
	target_t update(const target_t& mt_cur, const target_t& mt_grad)
	{
		return mt_cur - lr * mt_grad;
	}

	gd(const double& lr_i = 0.001) :lr(lr_i)
	{}

	void update_inert() 
	{}
};

template<>
struct gd<double>
{
	using target_t = double;
	double lr;
	target_t update(const target_t& mt_cur, const target_t& mt_grad)
	{
		return mt_cur - lr * mt_grad;			// ����Ҳ�ܽ�����ͨ�ĸ���
	}

	gd(const double& lr_i = 0.001) :lr(lr_i)
	{}

	void update_inert()
	{}
};

template<typename target_t>
struct adam
{
	using type = typename target_t::type;
	int t;
	target_t mtv;
	type dvb;
	type dvbt;
	target_t mts;
	type dsb;
	type dsbt;
	type dep;
	type lr;

	target_t update(const target_t& mt_cur, const target_t& mt_grad)
	{
		type one(1.);
		if (t == 0)
		{
			mtv = mt_grad;
			mts = mt_grad;
			return mt_cur - lr * mtv;
		}
		t++;
		dsbt = dsbt * dsb;
		mts = (dsb * mts + (one - dsb) * mt_grad * mt_grad);
		auto mts_ = mts / (one - dsbt);

		dvbt = dvbt * dvb;
		mtv = (dvb * mtv + (one - dvb) * mt_grad);
		auto mtv_ = mtv / (one - dvbt);

		return mt_cur - (lr * mtv_) / (sqrtl(mts_) + dep);
	}

	adam(const type& lr_i = 0.001, const type& dvb_i = 0.9, const type& dsb_i = 0.999, const type& dep_i = 1e-8)
		:t(0), lr(lr_i), dvb(dvb_i), dvbt(dvb_i), dsb(dsb_i), dsbt(dsb_i), dep(dep_i)
	{}

	void update_inert()
	{
		//t = 1;
		//dsbt = dsb;
		//dvbt = dvb;
	}
};

template<>
struct adam<double>
{
	using type = double;
	using target_t = double;
	int t;
	target_t mtv;
	type dvb;
	type dvbt;
	target_t mts;
	type dsb;
	type dsbt;
	type dep;
	type lr;

	target_t update(const target_t& mt_cur, const target_t& mt_grad)
	{
		type one(1.);
		if (t == 0)
		{
			mtv = mt_grad;
			mts = mt_grad;
			return mt_cur - lr * mtv;
		}
		t++;
		dsbt = dsbt * dsb;
		mts = (dsb * mts + (one - dsb) * mt_grad * mt_grad);
		auto mts_ = mts / (one - dsbt);

		dvbt = dvbt * dvb;
		mtv = (dvb * mtv + (one - dvb) * mt_grad);
		auto mtv_ = mtv / (one - dvbt);

		return mt_cur - (lr * mtv_) / (sqrtl(mts_) + dep);
	}

	adam(const type& lr_i = 0.001, const type& dvb_i = 0.9, const type& dsb_i = 0.999, const type& dep_i = 1e-8)
		:t(0), lr(lr_i), dvb(dvb_i), dvbt(dvb_i), dsb(dsb_i), dsbt(dsb_i), dep(dep_i)
	{}

	void update_inert()
	{
		//t = 1;
		//dsbt = dsb;
		//dvbt = dvb;
	}
};

template<typename target_t>
struct nadam
{
	using type = typename target_t::type;
	int t;
	target_t mtv;
	type dvb;
	type dvbt;
	target_t mts;
	type dsb;
	type dsbt;
	type dep;
	type lr;

	target_t update(const target_t& mt_cur, const target_t& mt_grad)
	{
		type one(1.);
		if (t == 0)
		{
			mtv = mt_grad;
			mts = mt_grad;
			return mt_cur - lr * mtv;
		}
		t++;
		dsbt = dsbt * dsb;
		mts = (dsb * mts + (one - dsb) * mt_grad * mt_grad);
		auto mts_ = mts / (one - dsbt);

		dvbt = dvbt * dvb;
		mtv = (dvb * mtv + (one - dvb) * mt_grad);
		auto mtv_ = mtv / (one - dvbt);

		auto mtv_n = lr * (dvb * mtv_ / (one - dvbt * dvb) + (one - dvb) / (one - dvbt) * mt_grad);
		return mt_cur - mtv_n / (sqrtl(mts_) + dep);
	}

	nadam(const type& lr_i = 0.002, const type& dvb_i = 0.9, const type& dsb_i = 0.999, const type& dep_i = 1e-8)
		:t(0), lr(lr_i), dvb(dvb_i), dvbt(dvb_i), dsb(dsb_i), dsbt(dsb_i), dep(dep_i)
	{
	}

	void update_inert()
	{
		t = 0;
		dsbt = dsb;
		dvbt = dvb;
	}
};

template<>
struct nadam<double>
{
	using type = double;
	using target_t = double;
	int t;
	target_t mtv;
	type dvb;
	type dvbt;
	target_t mts;
	type dsb;
	type dsbt;
	type dep;
	type lr;

	target_t update(const target_t& mt_cur, const target_t& mt_grad)
	{
		type one(1.);
		if (t == 0)
		{
			mtv = mt_grad;
			mts = mt_grad;
			return mt_cur - lr * mtv;
		}
		t++;
		dsbt = dsbt * dsb;
		mts = (dsb * mts + (one - dsb) * mt_grad * mt_grad);
		auto mts_ = mts / (one - dsbt);

		dvbt = dvbt * dvb;
		mtv = (dvb * mtv + (one - dvb) * mt_grad);
		auto mtv_ = mtv / (one - dvbt);

		auto mtv_n = lr * (dvb * mtv_ / (one - dvbt * dvb) + (one - dvb) / (one - dvbt) * mt_grad);
		return mt_cur - mtv_n / (sqrtl(mts_) + dep);
	}

	nadam(const type& lr_i = 0.002, const type& dvb_i = 0.9, const type& dsb_i = 0.999, const type& dep_i = 1e-8)
		:t(0), lr(lr_i), dvb(dvb_i), dvbt(dvb_i), dsb(dsb_i), dsbt(dsb_i), dep(dep_i)
	{
		
	}

	void update_inert()
	{
		t = 0;
		dsbt = dsb;
		dvbt = dvb;
	}
};

#endif

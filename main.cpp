#include <iostream>
#include "base_function.hpp"

void test_base_ops()
{
    mat<3, 1, double> mt1 = { 1, 2, 3 };
    mat<1, 3, double> mt2 = { 4, 5, 6 };
    mat<3, 3, double> mt3 = mt1.dot(mt2);
    mt3.print();
    mat<3, 3, double> mt4 = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    mat<3, 3, double> mt5 = mt3 - mt4;
    mt5.print();
    mat<3, 3, double> mt6 = mt5 / 2.0;
    mt6.print();
    mat<3, 3, double> mt7 = mt6 * 2.0;
    mt7.print();
    mat<3, 3, double> mt8 = mt7 + 1.0;
    mt8.print();
    mat<3, 3, double> mt9 = {
        1, 0, 0,
        0, 2, 0,
        0, 0, 1
    };
    printf("det(mt9)=%lf\r\n", det(mt9));
}

#include "bp.hpp"
#include "activate_function.hpp"

void test_bp()
{
    using net_t = bp<double, 1, nadam, softmax, HeMean, 3, 10>;
    net_t net;
    using input_t = typename net_t::input_type;
    using ret_t = typename net_t::ret_type;
    input_t mt_input = { .1, .2, .3 };
    ret_t mt_expected = { 0, 1., 0, 0, 0, 0, 0, 0, 0, 0 };
    for (int i = 0; i < 50000; ++i)
    {
        ret_t mt_out = net.forward(mt_input);
        net.backward(mt_out - mt_expected );
        net.update_inert();
    }
    net.forward(mt_input).print();
}

#include "restricked_boltzman_machine.hpp"

void test_rbm()
{
    using net_t = restricked_boltzman_machine<3, 16>;

    net_t rbm;
    using input_t = mat<3, 1, double>;
    auto train = {
        input_t{ 1, 0, 1 },
        input_t{ 0, 0, 1 },
        input_t{ 1, 0, 0 },
        input_t{ 1, 1, 1 },
        input_t{ 0, 0, 0 }
    };
    // 经过试验1W次训练可以让rbm记住这些数据
    for (int i = 0; i < 10000; ++i)
    {
        for (auto&& mt_input : train)
        {
            rbm.train(mt_input);
        }
    }
    for (auto&& mt_input : train)
    {
        auto mt_out = rbm.association(mt_input*0.8);
        mt_out.print();
    }
}

#include <random>
#include "gmm_t.hpp"

template<int dim_size>
mat<dim_size, dim_size, double> random_pos_def(std::default_random_engine& ge, double mean = 0.0, double stddev = 1.0) {
    std::normal_distribution<double> dist(mean, stddev);
    mat<dim_size, dim_size, double> A;
    for (int i = 0; i < dim_size; ++i)
        for (int j = 0; j < dim_size; ++j)
            A.get(i, j) = dist(ge);
    // S = A * A^T
    return A.dot(A.t());
}

void test_gmm()
{
    // 测试高斯混合模型
    constexpr int dim_size = 2;
	using mat_type = mat<dim_size, 1, double>;
	std::vector<mat_type> vec_x;

	/* 生成数据 */
	std::default_random_engine ge;
	std::normal_distribution <double> ud(0., 5.);
	const int set_size = 10;
	for (int i = 0; i < set_size; ++i)
	{
		mat_type mt;
		for (int i = 0; i < mt.size(); ++i) 
		{
			mt.get(i, 0) = ud(ge);
		}
		vec_x.push_back(mt);
	}
	std::normal_distribution <double> ud1(20., 8.);
	for (int i = 0; i < 2* set_size; ++i)
	{
		mat_type mt;
		for (int i = 0; i < dim_size; ++i)
		{
			mt.get(i, 0) = ud1(ge);
		}
		vec_x.push_back(mt);
	}
	std::normal_distribution <double> ud2(30., 4.);
	for (int i = 0; i < 2* set_size; ++i)
	{
		mat_type mt;
		for (int i = 0; i < dim_size; ++i)
		{
			mt.get(i, 0) = ud2(ge);
		}
		vec_x.push_back(mt);
	}
	/* 初始化类别 */
	std::random_device rd;
	std::mt19937 rng(rd());
	std::shuffle(vec_x.begin(), vec_x.end(), rng);

	std::vector<struct em_class<dim_size>> vec_cls(3);
	class def_init;
	for (int lp = 0; lp < vec_cls.size(); ++lp)
	{
        mat<dim_size, 1, double> A;
		weight_initilizer<def_init>::cal(vec_cls[lp].u, 0., 50.);           // 随机分配0-50之间的均值
        vec_cls[lp].sigma = random_pos_def<dim_size>(ge, 0., 10.); 
		vec_cls[lp].p = (1./vec_cls.size());                                // 初始化为相同权重
	}
	/* 执行分类 */
	gmm(vec_x, vec_cls);
	/* 展示分类结果 */
	for (int lp = 0; lp < vec_cls.size(); ++lp) 
	{
		printf("---------- %d -----------\r\n", lp);
		vec_cls[lp].u.print();
		vec_cls[lp].sigma.print();
		printf("p=%lf\r\n", vec_cls[lp].p);
	}
}

/* 测试决策树 */
#include "decision_tree.hpp"
#include "cart_t.hpp"

int cc(const mat<4,1,double>& m) 
{
	return m[3];
}

int pc(const int& idx, const mat<4, 1, double>& d)
{
	if (idx == 0) return d[0];
	if (idx == 1) return d[1];
	if (idx == 2) return d[2];
	return -1;
}

void test_decision_tree()
{
	std::vector<mat<4, 1, double> > vec_dat;

	vec_dat.push_back({ -1,  1,	-1, -1 });
	vec_dat.push_back({  1,	-1,  1,	 1 });
	vec_dat.push_back({  1,	-1, -1,  1 });
	vec_dat.push_back({ -1, -1, -1, -1 });
	vec_dat.push_back({ -1, -1,  1,	 1 });
	vec_dat.push_back({ -1,  1,	 1,	-1 });
	vec_dat.push_back({  1,	 1,	 1,	-1 });
	vec_dat.push_back({  1,	-1, -1,  1 });
	vec_dat.push_back({ -1,  1,	-1, -1 });

	struct dt_node* p_id3_tree = gen_id3_tree<3>(vec_dat, pc, cc);
	struct dt_node* p_c45_tree = gen_c45_tree<3>(vec_dat, pc, cc);
	struct dt_node* p_cart_tree = gen_cart_tree<3>(vec_dat, pc, cc);

	for (auto itr = vec_dat.begin(); itr != vec_dat.end(); ++itr) 
	{
		std::tuple<int,double> tp_id3 = judge_id3(p_id3_tree, *itr, pc, -2);
		std::tuple<int,double> tp_c45 = judge_c45(p_c45_tree, *itr, pc, -2);
		std::tuple<int,double> tp_cart = judge_cart(p_cart_tree, *itr, pc, -2);
		printf("ID3:%d\trate:%.2lf\tC4.5:%d\trate:%.2lf\tCART:%d\trate:%.2lf\tLABEL:%d\r\n"
			, std::get<0>(tp_id3), std::get<1>(tp_id3)
			, std::get<0>(tp_c45), std::get<1>(tp_c45)
			, std::get<0>(tp_cart), std::get<1>(tp_cart)
			, cc((*itr)[3]));
	}
}

/* 测试DBN */
#include <iomanip>
#include "dbn_t.hpp"
#include <termios.h>
#include <unistd.h>

int _getch() {
    struct termios oldt, newt;
    int ch;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    ch = getchar();
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    return ch;
}

struct train_data
{
	mat<28, 28, double> mt_image;
	mat<10, 1, double> mt_label;
	int					i_num;
};

void assign_mat(mat<28, 28, double>& mt, const unsigned char* sz)
{
	int i_sz_cnt = 0;
	for (int r = 0; r < 28; ++r)
	{
		for (int c = 0; c < 28; ++c)
		{
			mt.get(r, c) = sz[i_sz_cnt++];
		}
	}
}

void test_dbn()
{
	unsigned char sz_image_buf[28 * 28];
	std::vector<train_data> vec_train_data;
	ht_memory mry_train_images(ht_memory::big_endian);
	mry_train_images.read_file("./data/train-images.idx3-ubyte");
	int32_t i_image_magic_num = 0, i_image_num = 0, i_image_col_num = 0, i_image_row_num = 0;
	mry_train_images >> i_image_magic_num >> i_image_num >> i_image_row_num >> i_image_col_num;
	printf("magic num:%d | image num:%d | image_row:%d | image_col:%d\r\n"
		, i_image_magic_num, i_image_num, i_image_row_num, i_image_col_num);
	ht_memory mry_train_labels(ht_memory::big_endian);
	mry_train_labels.read_file("./data/train-labels.idx1-ubyte");
	int32_t i_label_magic_num = 0, i_label_num = 0;
	mry_train_labels >> i_label_magic_num >> i_label_num;
	for (int i = 0; i < i_image_num; ++i)
	{
		memset(sz_image_buf, 0, sizeof(sz_image_buf));
		train_data td;
		unsigned char uc_label = 0;
		mry_train_images.read((char*)sz_image_buf, sizeof(sz_image_buf));
		assign_mat(td.mt_image, sz_image_buf);
		td.mt_image = td.mt_image / 256.;
		mry_train_labels >> uc_label;
		td.i_num = uc_label;
		td.mt_label.get((int)uc_label, 0) = 1;
		vec_train_data.push_back(td);
	}
	std::random_device rd;
	std::mt19937 rng(rd());
	std::shuffle(vec_train_data.begin(), vec_train_data.end(), rng);

	using dbn_type = dbn_t<double, 28 * 28, 10, 10, 10>;
	using mat_type = mat<28 * 28, 1, double>;
	using ret_type = dbn_type::ret_type;
	dbn_type dbn_net;
	std::vector<mat_type> vec_input;
	std::vector<ret_type> vec_expect;
	for (int i = 0; i < 50; ++i) 
	{
		vec_input.push_back(vec_train_data[i].mt_image.one_col());
		vec_expect.push_back(vec_train_data[i].mt_label.one_col()); 
	}
	dbn_net.pretrain(vec_input, 10000);
	dbn_net.finetune(vec_expect, 10000);
	while (1)
	{
		std::string str_test_num;
		std::cout << "#";
		std::getline(std::cin, str_test_num);
		int i_test_num = 0;
		try{
			i_test_num = std::stoi(str_test_num);
		}catch (std::exception& e)
		{
			std::cout << "Invalid input. Please enter a number." << std::endl;
			continue;
		}
		auto pred = dbn_net.forward(vec_train_data[i_test_num].mt_image.one_col());
		pred.print();
		vec_train_data[i_test_num].mt_label.one_col().print();
		std::cout << "Press 'q' to quit or any other key to continue..." << std::endl;
		if ('q' == _getch())break;
	}
}

#include "cascade_judger.hpp"

void test_cascade_judger()
{
	// 测试级联分类器
	// 1. 训练DBN
	// 2. 训练决策树
	// 3. 预测
	constexpr int i_data_num = 5;
	std::vector<market_data<i_data_num>> vec_data;
	cascade_judger_t<i_data_num> cj;
	cj.train(vec_data, 100, 100);
	// 4. 预测
	market_data<i_data_num> data;
	data.label = 0;
	double d_poss = 0.0;
	int i_ret = cj.predict(data, d_poss);
	std::cout << "Predicted class: " << i_ret << ", Possibility: " << d_poss << std::endl;
	// 5. 评估
	// 6. 保存模型
	// 7. 加载模型
}

#include "mha_t.hpp"

void test_mha()
{
	using namespace mha;
	using net_type = mha_t<3, 2, 4, double>;
	net_type mha_net;
	net_type::input_type mt_input = { 1, 2, 3, 4, 5, 6 };
	net_type::input_type mt_expect = {6,5,4,3,2,1};
	for (int i = 0; i < 100; ++i)
	{
		auto mt_out = mha_net.forward(mt_input);
		mha_net.backward((mt_out - mt_expect));
		mha_net.update_inert();
	}
	mha_net.forward(mt_input).print();
	ht_memory mry(system_endian());
	write_file(mha_net, mry);
	std::cout << "MHA test completed." << std::endl;
	net_type mha_net2;
	read_file(mry, mha_net2);
	mha_net2.forward(mt_input).print();
	std::cout << "MHA test completed with read_file." << std::endl;
}

int main(int argc, char** argv)
{
    //test_base_ops();
    //test_rbm();
    //test_gmm();
    //test_decision_tree();
	//test_dbn();
	test_cascade_judger();
	test_mha();
    return 0;
}
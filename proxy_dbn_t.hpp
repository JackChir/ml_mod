#ifndef __PROXY_DBN_T_HPP__
#define __PROXY_DBN_T_HPP__

#include <thread>
#include <vector>

#include "bp.hpp"
#include "dbn_t.hpp"

// DBN中的RBM进行学习，然后通过多个bp神经网络进行微调
template<int predict_num, typename val_t, int first_input_row, int...args>
struct predict_net_t
{
    using bp_type = bp<val_t, 1, nadam, ReLu, XavierGaussian, first_input_row, args...>;
    using input_type = typename bp_type::input_type;
    using ret_type = mat<bp_type::ret_type::r, predict_num, val_t>;   // 输出结果的类型
    using softmax_type = bp<val_t, 1, nadam, softmax, XavierGaussian, bp_type::ret_type::r, 200>; // Softmax层

    bp_type m_bps[predict_num];    // 每个预测结果对应一个BP神经网络
    softmax_type m_softmax[predict_num];    // 每个预测结果对应一个Softmax层

    ret_type forward(const input_type& input)
    {
        ret_type ret;
        /*
        for (int i = 0; i < predict_num; ++i)
        {
            auto&& bp_out = m_softmax[i].forward(m_bps[i].forward(input));    // 前向传播
            for (int j = 0; j < bp_type::ret_type::r; ++j)
            {
                ret.get(j, i) = bp_out.get(j, 0);    // 将每个BP的输出结果存入ret
            }
        }
            */
        // 创建predict_num个线程来并行处理每个BP神经网络
        std::vector<std::thread> threads;
        for (int i = 0; i < predict_num; ++i)
        {
            threads.emplace_back([&, i]() {
                auto&& bp_out = m_softmax[i].forward(m_bps[i].forward(input));    // 前向传播
                for (int j = 0; j < bp_type::ret_type::r; ++j)
                {
                    ret.get(j, i) = bp_out.get(j, 0);    // 将每个BP的输出结果存入ret
                }
            });
        }
        // 等待所有线程完成
        for (auto& th : threads)
        {
            if (th.joinable())
            {
                th.join();
            }
        }
        return ret;
    }

    input_type backward(const ret_type& ret)
    {
        /*
        input_type input;
        for (int i = 0; i < predict_num; ++i)
        {
            auto&& bp_out = m_bps[i].backward(m_softmax[i].backward(ret.col(i)));    // 反向传播
            for (int j = 0; j < bp_type::input_type::r; ++j)
            {
                input.get(j, 0) += bp_out.get(j, 0);    // 将每个BP的输入结果累加到input
            }
        }
        input = input / static_cast<val_t>(predict_num);    // 平均化输入
            */
        input_type deltas[predict_num];
        // 创建predict_num个线程来并行处理每个BP神经网络的反向传播
        std::vector<std::thread> threads;
        for (int i = 0; i < predict_num; ++i)
        {
            threads.emplace_back([&, i]() {
                deltas[i] = m_bps[i].backward(m_softmax[i].backward(ret.col(i)));    // 反向传播
            });
        }
        // 等待所有线程完成
        for (auto& th : threads)
        {
            if (th.joinable())
            {
                th.join();
            }
        }
        input_type delta;
        for (int i = 0; i < predict_num; ++i)
        {
            for (int j = 0; j < bp_type::input_type::r; ++j)
            {
                delta.get(j, 0) += deltas[i].get(j, 0);    // 将每个BP的输入结果累加到delta
            }
        }
        delta = delta / static_cast<val_t>(predict_num);    // 平均化输入
        return delta;
    }
};

template<int predict_num, typename val_t, int...args>
void write_file(const predict_net_t<predict_num, val_t, args...>& net, ht_memory& mry)
{
    for (int i = 0; i < predict_num; ++i)
    {
        write_file(net.m_bps[i], mry);    // 写入每个BP神经网络
        write_file(net.m_softmax[i], mry);    // 写入每个Softmax层
    }
}

template<int predict_num, typename val_t, int...args>
void read_file(predict_net_t<predict_num, val_t, args...>& net, ht_memory& mry)
{
    for (int i = 0; i < predict_num; ++i)
    {
        read_file(mry, net.m_bps[i]);    // 读取每个BP神经网络
        read_file(mry, net.m_softmax[i]);    // 读取每个Softmax层
    }
}

struct predict_result
{
    int idx;            // 预测结果的索引
    double d_poss;     // 预测的概率
};

template<typename trans_name, typename raw_data_type, int output_num>
class proxy_dbn_t
{
private:
    using local_trans_t = trans_t<trans_name, raw_data_type>;
    template<int ih>
    using predict_t = predict_net_t<output_num, double, ih, 200, 200, 200>;
    using dbn_type = dbn_t<predict_t, double, local_trans_t::input_size, local_trans_t::input_size/2, local_trans_t::input_size/4>;
    using input_type = typename dbn_type::input_type;
    using ret_type = typename dbn_type::ret_type;
    dbn_type m_dbn;    // 定义DBN模型

public:
    void train(const std::vector<raw_data_type>& vec_data, const int& i_pretrain_times = 100, const int& i_finetune_times = 100, const bool& sample = true)
    {
        std::vector<input_type> vec_input;
        vec_input.resize(vec_data.size());
        for (int idx = 0; idx < vec_data.size(); ++idx)
        {
            auto&& data = vec_data[idx];
            vec_input[idx] = local_trans_t::trans_data_type(data);    // 将RSI和盘口数据拼接
        }
        m_dbn.pretrain(vec_input, i_pretrain_times, sample);    // 预训练
        // 获得期望值，对DBN进行微调
        std::vector<ret_type> vec_expect;
        vec_expect.resize(vec_data.size());
        for (int idx = 0; idx < vec_data.size(); ++idx)
        {
            auto& data = vec_data[idx];
            auto& mt_expect = vec_expect[idx];
            mt_expect = 0.0;
            for (int i = 0; i < output_num; ++i)
            {
                int label = data.labels[i];
                mt_expect.get(label, i) = 1.0;    // 设置标签位置为1.0
            }
        }
        m_dbn.template finetune<cross_entropy>(vec_expect, i_finetune_times);    // 使用交叉熵损失函数作为损失函数进行微调
    }

    // 获取最大值的索引
    static int get_max_index(const mat<200, 1, double>& mt_out, double& d_poss)
    {
        int idx = 0;
        d_poss = mt_out.get(0, 0);
        for (int i = 1; i < 200; ++i)
        {
            if (mt_out.get(i, 0) > d_poss)
            {
                d_poss = mt_out.get(i, 0);
                idx = i;
            }
        }
        return idx;
    }

    void predict(const raw_data_type& raw_data, std::vector<predict_result>& vec_result, const bool& sample = true)
    {
        input_type data = local_trans_t::trans_data_type(raw_data);    // 将RSI和盘口数据拼接
        auto mt_out = m_dbn.forward(data, sample);    // 直接前向传播
        for (int c = 0; c < output_num; ++c)
        {
            predict_result result;
            result.idx = get_max_index(mt_out.col(c), result.d_poss);    // 获取最大值的索引和概率
            vec_result.push_back(result);    // 将结果添加到结果向量中
        }
    }
};


#endif

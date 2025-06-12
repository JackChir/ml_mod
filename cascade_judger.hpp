#ifndef _CASCADE_JUDGER_H_
#define _CASCADE_JUDGER_H_
/** 
 * @file cascade_judger.h
 * @brief 级联分类器的判断器
 * @details
 * 使用决策树结合DBN构建级联分类器，DBN分别基于1/5/10分钟RSI、MACD、KDJ、十档盘口、成交量等数据进行训练，生成结果为200维度的一个向量(涨跌10%，精确到0.1%就是200个档位)，用于指明上涨下跌的幅度；
 * 1、收集一段时间内的数据；
 * 2、使用收集到的数据对DBN进行一定次数的训练；
 * 3、使用数据对决策树进行训练；
 */

 #include "decision_tree.hpp"



#include "cart_t.hpp"

template<int data_num>
class cascade_judger_t
{
private:
    proxy_dbn_t<rsi_pv, market_data<data_num>> m_dbn_rsi_pv;
    proxy_dbn_t<macd_pv, market_data<data_num>> m_dbn_macd_pv;
    proxy_dbn_t<kdj_pv, market_data<data_num>> m_dbn_kdj_pv;
    proxy_dbn_t<macd_rsi, market_data<data_num>> m_dbn_macd_rsi;
    proxy_dbn_t<all_idx, market_data<data_num>> m_dbn_all;    // 所有指标的DBN
    dt_node* m_p_decision_tree;    // 决策树
private:
    void train_dbn(const std::vector<market_data<data_num>>& vec_data, const int& i_pretrain_times = 100, const int& i_finetune_times = 100)
    {
        m_dbn_rsi_pv.train(vec_data, i_pretrain_times, i_finetune_times);
        m_dbn_macd_pv.train(vec_data, i_pretrain_times, i_finetune_times);
        m_dbn_kdj_pv.train(vec_data, i_pretrain_times, i_finetune_times);
        m_dbn_macd_rsi.train(vec_data, i_pretrain_times, i_finetune_times);
        m_dbn_all.train(vec_data, i_pretrain_times, i_finetune_times);
    }

    int dbn_predict(const int& idx, const market_data<data_num>& raw_data, double& d_poss)
    {
        // 先使用RSI+盘口数据进行判断
        if (idx == 0)
            return m_dbn_rsi_pv.predict(raw_data, d_poss);

        // 再使用MACD+盘口数据进行判断
        if (idx == 1)
            return m_dbn_macd_pv.predict(raw_data, d_poss);

        // 再使用KDJ+盘口数据进行判断
        if (idx == 2)
            return m_dbn_kdj_pv.predict(raw_data, d_poss);

        // 再使用MACD+RSI进行判断
        if (idx == 3)
            return m_dbn_macd_rsi.predict(raw_data, d_poss);

        // 最后使用所有指标进行判断
        if (idx == 4)
            return m_dbn_all.predict(raw_data, d_poss);
        
        return 0;           // 这里不会执行到，除非判断期的数量设置错误
    }

    void train_dt(const std::vector<market_data<data_num>>& vec_data, const double& stop_rate = 0.7)
    {
        // 训练决策树
        using dt_node_t = dt_node;
        std::function<int(const int&, const market_data<data_num>&)> pc = [this](const int& idx, const market_data<data_num>& d) {
            double d_poss = 0.0;         // DBN判断出的概率
            return this->dbn_predict(idx, d, d_poss);
        };
        m_p_decision_tree = gen_cart_tree<5>(vec_data, pc, get_market_data_label<data_num>, stop_rate);    // 生成决策树
    }
public:

    cascade_judger_t()
        : m_p_decision_tree(nullptr)
    {
    }
    ~cascade_judger_t()
    {
        if (m_p_decision_tree != nullptr)
        {
            delete m_p_decision_tree;
            m_p_decision_tree = nullptr;
        }
    }

    void train(const std::vector<market_data<data_num>>& vec_data, const int& i_pretrain_times = 100, const int& i_finetune_times = 100, const double& stop_rate = 0.7)
    {
        if (vec_data.empty())
        {
            std::cerr << "Error: No data to train." << std::endl;
            return;
        }
        train_dbn(vec_data, i_pretrain_times, i_finetune_times);
        train_dt(vec_data, stop_rate);
    }

    int predict(const market_data<data_num>& raw_data, double& d_poss)
    {
        if (m_p_decision_tree == nullptr)
        {
            std::cerr << "Error: Decision tree is not trained." << std::endl;
            return -1;  // 未训练决策树
        }
        // 使用决策树进行判断
        std::function<int(const int&, const market_data<data_num>&)> pc = [this, &d_poss](const int& idx, const market_data<data_num>& d) {
            double d_poss = 0.0;         // DBN判断出的概率
            return this->dbn_predict(idx, d, d_poss);
        };
        int i_label = 0;
        std::tuple<int, double> tp = judge_cart(m_p_decision_tree, raw_data, pc, -1);
        printf("Decision Tree: label=%d, rate=%.2lf\r\n", std::get<0>(tp), std::get<1>(tp));
        i_label = std::get<0>(tp);
        return i_label;
    }

    void write_file(ht_memory& mry) const
    {
        // 将DBN和决策树写入内存
        write_file(m_dbn_rsi_pv, mry);    // 写入RSI+盘口数据的DBN
        write_file(m_dbn_macd_pv, mry);    // 写入MACD+盘口数据的DBN
        write_file(m_dbn_kdj_pv, mry);    // 写入KDJ+盘口数据的DBN
        write_file(m_dbn_macd_rsi, mry);    // 写入MACD+RSI的DBN
        write_file(m_dbn_all, mry);    // 写入所有指标的DBN
        write_file(m_p_decision_tree, mry);    // 写入决策树
    }

    void read_file(ht_memory& mry)
    {
        // 从内存中读取DBN和决策树
        read_file(mry, m_dbn_rsi_pv);    // 读取RSI+盘口数据的DBN
        read_file(mry, m_dbn_macd_pv);    // 读取MACD+盘口数据的DBN
        read_file(mry, m_dbn_kdj_pv);    // 读取KDJ+盘口数据的DBN
        read_file(mry, m_dbn_macd_rsi);    // 读取MACD+RSI的DBN
        read_file(mry, m_dbn_all);    // 读取所有指标的DBN
        read_file(mry, m_p_decision_tree);    // 读取决策树
    }

};

#endif

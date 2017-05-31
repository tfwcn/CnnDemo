using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace EmguCVDemo.BP
{
    public class AnnClassifier
    {
        private int mInputCount;
        private int mHiddenCount;
        private int mOutputCount;

        private List<NetworkNode> mInputNodes;
        private List<NetworkNode> mHiddenNodes;
        private List<NetworkNode> mOutputNodes;

        private float[,] mInputHiddenWeight;
        private float[,] mHiddenOutputWeight;

        private List<DataNode> trainNodes;

        public void setTrainNodes(List<DataNode> trainNodes)
        {
            this.trainNodes = trainNodes;
        }

        public AnnClassifier(int inputCount, int hiddenCount, int outputCount)
        {
            trainNodes = new List<DataNode>();
            mInputCount = inputCount;
            mHiddenCount = hiddenCount;
            mOutputCount = outputCount;
            mInputNodes = new List<NetworkNode>();
            mHiddenNodes = new List<NetworkNode>();
            mOutputNodes = new List<NetworkNode>();
            mInputHiddenWeight = new float[inputCount, hiddenCount];
            mHiddenOutputWeight = new float[mHiddenCount, mOutputCount];
        }

        /**
         * 更新权重，每个权重的梯度都等于与其相连的前一层节点的输出乘以与其相连的后一层的反向传播的输出
         */
        private void updateWeights(float eta)
        {
            // 更新输入层到隐层的权重矩阵
            for (int i = 0; i < mInputCount; i++)
                for (int j = 0; j < mHiddenCount; j++)
                    mInputHiddenWeight[i, j] -= eta
                            * mInputNodes[i].getForwardOutputValue()
                            * mHiddenNodes[j].getBackwardOutputValue();
            // 更新隐层到输出层的权重矩阵
            for (int i = 0; i < mHiddenCount; i++)
                for (int j = 0; j < mOutputCount; j++)
                    mHiddenOutputWeight[i, j] -= eta
                            * mHiddenNodes[i].getForwardOutputValue()
                            * mOutputNodes[j].getBackwardOutputValue();
        }

        /**
         * 前向传播
         */
        private void forward(List<float> list)
        {
            // 输入层
            for (int k = 0; k < list.Count(); k++)
                mInputNodes[k].setForwardInputValue(list[k]);
            // 隐层
            for (int j = 0; j < mHiddenCount; j++)
            {
                float temp = 0;
                for (int k = 0; k < mInputCount; k++)
                    temp += mInputHiddenWeight[k, j]
                            * mInputNodes[k].getForwardOutputValue();
                mHiddenNodes[j].setForwardInputValue(temp);
            }
            // 输出层
            for (int j = 0; j < mOutputCount; j++)
            {
                float temp = 0;
                for (int k = 0; k < mHiddenCount; k++)
                    temp += mHiddenOutputWeight[k, j]
                            * mHiddenNodes[k].getForwardOutputValue();
                mOutputNodes[j].setForwardInputValue(temp);
            }
        }

        /**
         * 反向传播
         */
        private void backward(int type)
        {
            // 输出层
            for (int j = 0; j < mOutputCount; j++)
            {
                // 输出层计算误差把误差反向传播，这里-1代表不属于，1代表属于
                float result = -1;
                if (j == type)
                    result = 1;
                mOutputNodes[j].setBackwardInputValue(
                        mOutputNodes[j].getForwardOutputValue() - result);
            }
            // 隐层
            for (int j = 0; j < mHiddenCount; j++)
            {
                float temp = 0;
                for (int k = 0; k < mOutputCount; k++)
                    temp += mHiddenOutputWeight[j, k]
                            * mOutputNodes[k].getBackwardOutputValue();
                mHiddenNodes[j].setBackwardInputValue(temp);
            }
        }

        public void train(float eta, int n)
        {
            reset();
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < trainNodes.Count(); j++)
                {
                    forward(trainNodes[j].getAttribList());
                    backward(trainNodes[j].getType());
                    updateWeights(eta);
                }
                System.Console.WriteLine("n = " + i);

            }
        }

        /**
         * 初始化
         */
        private void reset()
        {
            mInputNodes.Clear();
            mHiddenNodes.Clear();
            mOutputNodes.Clear();
            for (int i = 0; i < mInputCount; i++)
                mInputNodes.Add(new NetworkNode(NetworkNode.TYPE_INPUT));
            for (int i = 0; i < mHiddenCount; i++)
                mHiddenNodes.Add(new NetworkNode(NetworkNode.TYPE_HIDDEN));
            for (int i = 0; i < mOutputCount; i++)
                mOutputNodes.Add(new NetworkNode(NetworkNode.TYPE_OUTPUT));
            for (int i = 0; i < mInputCount; i++)
                for (int j = 0; j < mHiddenCount; j++)
                    mInputHiddenWeight[i, j] = (float)(new Random().NextDouble() * 0.1);
            for (int i = 0; i < mHiddenCount; i++)
                for (int j = 0; j < mOutputCount; j++)
                    mHiddenOutputWeight[i, j] = (float)(new Random().NextDouble() * 0.1);
        }

        public int test(DataNode dn)
        {
            forward(dn.getAttribList());
            float result = 2;
            int type = 0;
            // 取最接近1的
            for (int i = 0; i < mOutputCount; i++)
                if ((1 - mOutputNodes[i].getForwardOutputValue()) < result)
                {
                    result = 1 - mOutputNodes[i].getForwardOutputValue();
                    type = i;
                }
            return type;
        }
    }
}

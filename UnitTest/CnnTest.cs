using CnnDemo.CNN;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;

namespace UnitTest
{


    /// <summary>
    ///这是 CnnTest 的测试类，旨在
    ///包含所有 CnnTest 单元测试
    ///</summary>
    [TestClass()]
    public class CnnTest
    {


        private TestContext testContextInstance;

        /// <summary>
        ///获取或设置测试上下文，上下文提供
        ///有关当前测试运行及其功能的信息。
        ///</summary>
        public TestContext TestContext
        {
            get
            {
                return testContextInstance;
            }
            set
            {
                testContextInstance = value;
            }
        }

        #region 附加测试特性
        // 
        //编写测试时，还可使用以下特性:
        //
        //使用 ClassInitialize 在运行类中的第一个测试前先运行代码
        //[ClassInitialize()]
        //public static void MyClassInitialize(TestContext testContext)
        //{
        //}
        //
        //使用 ClassCleanup 在运行完类中的所有测试后再运行代码
        //[ClassCleanup()]
        //public static void MyClassCleanup()
        //{
        //}
        //
        //使用 TestInitialize 在运行每个测试前先运行代码
        //[TestInitialize()]
        //public void MyTestInitialize()
        //{
        //}
        //
        //使用 TestCleanup 在运行完每个测试后运行代码
        //[TestCleanup()]
        //public void MyTestCleanup()
        //{
        //}
        //
        #endregion


        /// <summary>
        ///TrainFullLayer 的测试
        ///</summary>
        [TestMethod()]
        public void TrainFullLayerTest()
        {
            Cnn target = new Cnn(); // TODO: 初始化为适当的值
            target.AddCnnFullLayer(100, 14, CnnNeuron.ActivationFunctionTypes.Tanh, false);
            target.AddCnnFullLayer(10, CnnNeuron.ActivationFunctionTypes.Tanh, false);
            double[] input = new double[100]; // TODO: 初始化为适当的值
            TestMethods.InitRandom(input);
            double[] output = new double[10]; // TODO: 初始化为适当的值
            TestMethods.InitRandom(output);
            double learningRate = 0.01F; // TODO: 初始化为适当的值
            double[] expected = null; // TODO: 初始化为适当的值
            double[] actual;
            double[] forwardOutputFullExpected = null; // TODO: 初始化为适当的值
            double[] forwardOutputFull = target.PredictFullLayer(input);//前向传播
            //计算输出误差
            double[] residual = new double[10]; // TODO: 初始化为适当的值
            for (int i = 0; i < 10; i++)
                residual[i] = output[i] - forwardOutputFull[i];
            expected = target.TrainFullLayer(input, output, learningRate, ref forwardOutputFullExpected);//前向传播+反向传播
            //actual = target.TrainFullLayer(residual, learningRate);//反向传播
            //Assert.AreEqual(expected, actual);
            //Assert.Inconclusive("验证此测试方法的正确性。");
            Assert.IsTrue(TestMethods.ArrayEqual(forwardOutputFullExpected, forwardOutputFull), "前向传播");
            //Assert.IsTrue(TestMethods.ArrayEqual(expected, actual), "反向传播");//反向传播无法测试
        }

        /// <summary>
        ///Train 的测试
        ///</summary>
        [TestMethod()]
        public void TrainTest()
        {
            Cnn target = new Cnn(); // TODO: 初始化为适当的值
            target.AddCnnConvolutionLayer(6, 250, 250, 10, 10, 1, 1, 1, 1, CnnNeuron.ActivationFunctionTypes.Tanh,
                4, 4, CnnPoolingNeuron.PoolingTypes.MaxPooling, false);
            target.AddCnnConvolutionLayer(20, 5, 5, 1, 1, 1, 1, CnnNeuron.ActivationFunctionTypes.Tanh,
                4, 4, CnnPoolingNeuron.PoolingTypes.MaxPooling, false, false);
            target.AddCnnConvolutionLayer(60, 5, 5, 1, 1, 1, 1, CnnNeuron.ActivationFunctionTypes.Tanh,
                2, 2, CnnPoolingNeuron.PoolingTypes.MeanPooling, false, false);
            target.AddCnnFullLayer(50, CnnNeuron.ActivationFunctionTypes.Tanh, false);
            double[,] input = new double[250, 250]; // TODO: 初始化为适当的值
            TestMethods.InitRandom(input);
            double[] output = new double[50]; // TODO: 初始化为适当的值
            TestMethods.InitRandom(output);
            double learningRate = 0.01F; // TODO: 初始化为适当的值
            double[] forwardOutputFullExpected = null; // TODO: 初始化为适当的值
            List<List<double[,]>> expected = null; // TODO: 初始化为适当的值
            List<List<double[,]>> actual;
            double[] forwardOutputFull = target.Predict(input);//前向传播
            //计算输出误差
            double[] residual = new double[50]; // TODO: 初始化为适当的值
            for (int i = 0; i < 50; i++)
                residual[i] = output[i] - forwardOutputFull[i];
            expected = target.Train(input, output, learningRate, ref forwardOutputFullExpected);//前向传播+反向传播
            //actual = target.Train(residual, learningRate);//反向传播
            //Assert.AreEqual(forwardOutputFullExpected, forwardOutputFull);
            //Assert.AreEqual(expected, actual);
            //Assert.Inconclusive("验证此测试方法的正确性。");
            Assert.IsTrue(TestMethods.ArrayEqual(forwardOutputFullExpected, forwardOutputFull), "前向传播");
            //for (int i = 0; i < expected.Count; i++)
            //{
            //    for (int j = 0; j < expected[i].Count; j++)
            //    {
            //        Assert.IsTrue(TestMethods.ArrayEqual(expected[i][j], actual[i][j]), "反向传播");//反向传播无法测试
            //    }
            //}
        }
    }
}

using CnnDemo.CNN;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using CnnDemo.CNN.Model;

namespace UnitTest
{


    /// <summary>
    ///这是 CnnHelperTest 的测试类，旨在
    ///包含所有 CnnHelperTest 单元测试
    ///</summary>
    [TestClass()]
    public class CnnHelperTest
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
        ///ConvolutionFull 的测试
        ///</summary>
        [TestMethod()]
        public void ConvolutionFullTest()
        {
            int shareWeightWidth = 10;//感知野宽
            int shareWeightHeight = 10;//感知野高
            int valueWidth = 250;//矩阵宽
            int valueHeight = 250;//矩阵高

            int receptiveFieldOffsetWidth = 1; // TODO: 初始化为适当的值
            int receptiveFieldOffsetHeight = 1; // TODO: 初始化为适当的值
            int offsetWidth = 1; // TODO: 初始化为适当的值
            int offsetHeight = 1; // TODO: 初始化为适当的值
            int kernelWidth = Convert.ToInt32(Math.Ceiling((valueWidth + offsetWidth - shareWeightWidth * receptiveFieldOffsetWidth) / (double)offsetWidth));//卷积核宽
            int kernelHeight = Convert.ToInt32(Math.Ceiling((valueHeight + offsetHeight - shareWeightHeight * receptiveFieldOffsetHeight) / (double)offsetHeight));//卷积核高
            double[,] receptiveField = new double[shareWeightWidth, shareWeightHeight]; // TODO: 初始化为适当的值
            TestMethods.InitRandom(receptiveField);
            double[,] value = new double[kernelWidth, kernelHeight]; //卷积核
            TestMethods.InitRandom(value);
            double[,] expected = null; //应该输出的正确值
            double[,] actual;

            expected = CnnHelperOld.ConvolutionFull(receptiveField, value);
            actual = CnnHelper.ConvolutionFull(receptiveField, receptiveFieldOffsetWidth, receptiveFieldOffsetHeight,
                value, offsetWidth, offsetHeight,
                valueWidth, valueHeight);
            //Assert.Inconclusive("验证此测试方法的正确性。");
            Assert.IsTrue(TestMethods.ArrayEqual(expected, actual));
        }

        /// <summary>
        ///ConvolutionValid 的测试
        ///</summary>
        [TestMethod()]
        public void ConvolutionValidTest()
        {
            int shareWeightWidth = 10;//感知野宽
            int shareWeightHeight = 10;//感知野高
            int valueWidth = 250;//矩阵宽
            int valueHeight = 250;//矩阵高
            int receptiveFieldOffsetWidth = 1; // TODO: 初始化为适当的值
            int receptiveFieldOffsetHeight = 1; // TODO: 初始化为适当的值
            int offsetWidth = 1; // TODO: 初始化为适当的值
            int offsetHeight = 1; // TODO: 初始化为适当的值
            //int kernelWidth = Convert.ToInt32(Math.Ceiling((valueWidth + offsetWidth - shareWeightWidth * receptiveFieldOffsetWidth) / (double)offsetWidth));//卷积核宽
            //int kernelHeight = Convert.ToInt32(Math.Ceiling((valueHeight + offsetHeight - shareWeightHeight * receptiveFieldOffsetHeight) / (double)offsetHeight));//卷积核高
            CnnPaddingSize paddingSize = null; // TODO: 初始化为适当的值
            double[,] receptiveField = new double[shareWeightWidth, shareWeightHeight]; // TODO: 初始化为适当的值
            TestMethods.InitRandom(receptiveField);
            double[,] value = new double[valueWidth, valueHeight]; //卷积核
            TestMethods.InitRandom(value);
            double[,] expected = null; // TODO: 初始化为适当的值
            double[,] actual;
            expected = CnnHelperOld.ConvolutionValid(receptiveField, value);
            actual = CnnHelper.ConvolutionValid(receptiveField, receptiveFieldOffsetWidth, receptiveFieldOffsetHeight,
                value, offsetWidth, offsetHeight, paddingSize);
            //Assert.Inconclusive("验证此测试方法的正确性。");
            Assert.IsTrue(TestMethods.ArrayEqual(expected, actual));
        }
    }
}

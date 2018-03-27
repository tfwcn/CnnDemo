using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace UnitTest
{
    public static class TestMethods
    {
        /// <summary>
        /// 随机-1到1赋值
        /// </summary>
        /// <param name="array"></param>
        public static void InitRandom(double[,] array)
        {
            Random random = new Random();
            for (int i = 0; i < array.GetLength(0); i++)
            {
                for (int j = 0; j < array.GetLength(1); j++)
                {
                    array[i, j] = random.NextDouble() * 2 - 1;
                }
            }
        }
        /// <summary>
        /// 随机-1到1赋值
        /// </summary>
        /// <param name="array"></param>
        public static void InitRandom(double[] array)
        {
            Random random = new Random();
            for (int i = 0; i < array.GetLength(0); i++)
            {
                array[i] = random.NextDouble() * 2 - 1;
            }
        }
        /// <summary>
        /// 数组比较
        /// </summary>
        /// <param name="array"></param>
        public static bool ArrayEqual(double[,] array1, double[,] array2)
        {
            if (array1.GetLength(0) != array2.GetLength(0) || array1.GetLength(1) != array2.GetLength(1))
                return false;
            for (int i = 0; i < array1.GetLength(0); i++)
            {
                for (int j = 0; j < array1.GetLength(1); j++)
                {
                    if (array1[i, j] != array2[i, j])
                    {
                        return false;
                    }
                }
            }
            return true;
        }
        /// <summary>
        /// 数组比较
        /// </summary>
        /// <param name="array"></param>
        public static bool ArrayEqual(double[] array1, double[] array2)
        {
            if (array1.GetLength(0) != array2.GetLength(0))
                return false;
            for (int i = 0; i < array1.GetLength(0); i++)
            {
                if (array1[i] != array2[i])
                {
                    return false;
                }
            }
            return true;
        }
    }
}

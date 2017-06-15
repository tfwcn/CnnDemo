using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace EmguCVDemo.BP
{
    public class CnnHelper
    {
        public static void ShowChange(double[] output, double[] labels)
        {
            double subVal = 0;
            for (int i = 0; i < output.Length; i++)
            {
                subVal += output[i] - labels[i];
            }
            Console.WriteLine("CnnChange:" + subVal);
        }
    }
}

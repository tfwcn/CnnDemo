using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CnnDemo.BP
{
    public class DataNode
    {
        private List<float> mAttribList;
        private int type;

        public int getType()
        {
            return type;
        }

        public void setType(int type)
        {
            this.type = type;
        }

        public double[] getTypes()
        {
            double[] t = new double[3];
            t[0] = -1;
            t[1] = -1;
            t[2] = -1;
            t[type] = 1;
            return t;
        }

        public List<float> getAttribList()
        {
            return mAttribList;
        }

        public List<double> getAttribList2()
        {
            List<double> t = new List<double>();
            foreach (var v in mAttribList)
            {
                t.Add(v);
            }
            return t;
        }

        public void addAttrib(float value)
        {
            mAttribList.Add(value);
        }

        public DataNode()
        {
            mAttribList = new List<float>();
        }
    }
}

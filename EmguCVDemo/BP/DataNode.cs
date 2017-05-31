using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace EmguCVDemo.BP
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

        public List<float> getAttribList()
        {
            return mAttribList;
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

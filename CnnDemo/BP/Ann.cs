using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CnnDemo.BP
{
    public class Ann
    {
        private Dictionary<int, AnnNode[]> networkNode = new Dictionary<int, AnnNode[]>();
        public void SetLayerSizes(int[] layerSizes, int[] layerType)
        {
            try
            {
                for (int i = 0; i < layerSizes.Length; i++)
                {
                    //初始化神经元
                    networkNode.Add(i, new AnnNode[layerSizes[i]]);
                    for (int j = 0; j < layerSizes[i]; j++)
                    {
                        networkNode[i][j] = new AnnNode();
                    }
                    //初始化权重
                    foreach (var node in networkNode[i])
                    {
                        if (i == 0)
                        {
                            node.InputWeight = new int[1];
                        }
                        else if (i == 1)
                        {
                            node.InputWeight = new int[layerSizes[i - 1]];
                        }
                        else
                        {
                            //node.InputWeight = new int[layerSizes[i - 1]];
                        }
                    }
                }
            }
            catch (Exception ex)
            {

            }
        }
    }
}

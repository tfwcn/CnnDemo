using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using TensorFlow;

namespace TensorFlowSharpDemo
{
    public partial class Form1 : Form
    {
        TFGraph graph;
        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            try
            {
                using (var session = new TFSession())
                {
                    graph = session.Graph;

                    var a = graph.Const(2);
                    var b = graph.Const(3);
                    Console.WriteLine("a=2 b=3");

                    // Add two constants
                    var addingResults = session.GetRunner().Run(graph.Add(a, b));
                    var addingResultValue = addingResults.GetValue();
                    Console.WriteLine("a+b={0}", addingResultValue);

                    // Multiply two constants
                    var multiplyResults = session.GetRunner().Run(graph.Mul(a, b));
                    var multiplyResultValue = multiplyResults.GetValue();
                    Console.WriteLine("a*b={0}", multiplyResultValue);
                }
            }
            catch (Exception ex)
            {

            }
        }
    }
}

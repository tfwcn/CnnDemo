using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.IO;
using System.Threading;

namespace CnnDemo
{
    public partial class Form5 : Form
    {
        public Form5()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            Thread t = new Thread(() =>
            {
                try
                {
                    if (!Directory.Exists("imagenet"))
                    {
                        Directory.CreateDirectory("imagenet");
                    }
                    using (StreamReader sr = new StreamReader("fall11_urls.txt"))
                    {
                        string line = sr.ReadLine();
                        while (line != null && !String.IsNullOrEmpty(line))
                        {
                            string name = line.Split('\t')[0];
                            string url = line.Split('\t')[1];
                            if (!File.Exists("imagenet\\" + name + ".jpg"))
                            {
                                this.Invoke(new Action(() =>
                                {
                                    lblUrl.Text = line;
                                }));
                                DownloadFile(url, "imagenet\\" + name + ".jpg");
                            }
                            line = sr.ReadLine();
                        }
                    }
                }
                catch (System.Exception ex)
                {
                    Console.WriteLine(ex.ToString());
                }
            });
            t.Start();
        }
        private bool DownloadFile(string URL, string filename)
        {
            try
            {
                System.Net.HttpWebRequest Myrq = (System.Net.HttpWebRequest)System.Net.HttpWebRequest.Create(URL);
                Myrq.Timeout = 5000;
                Myrq.ReadWriteTimeout = 5000;
                System.Net.HttpWebResponse myrp = (System.Net.HttpWebResponse)Myrq.GetResponse();
                System.IO.Stream st = myrp.GetResponseStream();
                System.IO.Stream so = new System.IO.FileStream(filename, System.IO.FileMode.Create);
                byte[] by = new byte[1024];
                int osize = st.Read(by, 0, (int)by.Length);
                while (osize > 0)
                {
                    so.Write(by, 0, osize);
                    osize = st.Read(by, 0, (int)by.Length);
                }
                so.Close();
                st.Close();
                myrp.Close();
                Myrq.Abort();
                return true;
            }
            catch (System.Exception e)
            {
                return false;
            }
        }
    }
}

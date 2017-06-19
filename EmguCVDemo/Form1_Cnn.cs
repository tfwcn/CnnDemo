using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.Util;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using Emgu.CV.ML;
using System.Reflection;
using System.Threading;
using System.IO;
using EmguCVDemo.BP;

namespace EmguCVDemo
{
    public partial class Form1_Cnn : Form
    {
        VideoCapture cap;
        //CascadeClassifier cascadeClassifier;
        Mat nextFrame = new Mat();
        Bitmap nowImg;
        //SVM svmObj = new SVM();
        Rectangle faceRectangle = new Rectangle();
        Rectangle faceReadRectangle = new Rectangle();
        List<Rectangle> faceReadRectangleList = new List<Rectangle>();
        string url = "";
        //string url = "rtsp://admin:admin123@192.168.3.64:554/h264/ch1/main/av_stream";
        //string url = "rtsp://admin:admin@192.168.0.199/0";
        public Form1_Cnn()
        {
            InitializeComponent();
            CreateBP();
        }

        private void Form1_Cnn_Load(object sender, EventArgs e)
        {
            //CvInvoke.UseOpenCL = false;
            // passing 0 gets zeroth webcam
            cap = new VideoCapture(url);
            cap.ImageGrabbed += new EventHandler(cap_ImageGrabbed);
            // adjust path to find your xml
            //cascadeClassifier = new CascadeClassifier("haarcascade_frontalface_alt2.xml");

            btnStart.Enabled = true;
            btnPause.Enabled = false;
            btnStop.Enabled = false;
        }

        private void cap_ImageGrabbed(object sender, EventArgs e)
        {
            GetFrame();
        }

        private void btnStart_Click(object sender, EventArgs e)
        {
            cap.Start();
            btnStart.Enabled = false;
            btnPause.Enabled = true;
            btnStop.Enabled = true;
        }

        private void btnPause_Click(object sender, EventArgs e)
        {
            cap.Pause();
            btnStart.Enabled = true;
            btnPause.Enabled = false;
            btnStop.Enabled = false;
        }

        private void btnStop_Click(object sender, EventArgs e)
        {
            cap.Stop();
            //cap.Dispose();
            //cap = null;
            //GC.Collect();
            //cap = new VideoCapture(url);
            //cap.ImageGrabbed += new EventHandler(cap_ImageGrabbed);
            btnStart.Enabled = true;
            btnPause.Enabled = false;
            btnStop.Enabled = false;
            //Console.WriteLine(String.Format("start:{0:yyyy-MM-dd HH:mm:ss.fff}", DateTime.Now));
            //Image<Bgr, Byte> img = new Image<Bgr, Byte>(nowImg);
            //// there's only one channel (greyscale), hence the zero index
            //var faces = cascadeClassifier.DetectMultiScale(
            //    img, 1.1, 3,
            //    new Size(80, 80),
            //    new Size(nextFrame.Width / 8, nextFrame.Height / 8)
            //    );

            //foreach (var face in faces)
            //{
            //    img.Draw(face, new Bgr(0, double.MaxValue, 0), 3);
            //}
            //pbImg.Image = nextFrame.Bitmap;
            nowImg = (Bitmap)pbImg.Image;
            //Console.WriteLine(String.Format("end:{0:yyyy-MM-dd HH:mm:ss.fff}", DateTime.Now));
        }

        private void btnTrain_Click(object sender, EventArgs e)
        {
        }

        private bool isDown = false;
        private void pbImg_MouseDown(object sender, MouseEventArgs e)
        {
            //获取时间图片大小
            PropertyInfo rectangleProperty = this.pbImg.GetType().GetProperty("ImageRectangle", BindingFlags.Instance | BindingFlags.NonPublic);
            Rectangle rectangle = (Rectangle)rectangleProperty.GetValue(this.pbImg, null);
            DrawRectangle(e.X - rectangle.X, e.Y - rectangle.Y);
            faceRectangle.X = e.X - rectangle.X;
            faceRectangle.Y = e.Y - rectangle.Y;
            //faceRectangle.Width = 0;
            //faceRectangle.Height = 0;
            DrawRectangle(e.X - rectangle.X, e.Y - rectangle.Y);
            DrawRectangleList();
            isDown = true;
            PrintMsg(String.Format("MouseDown:{0},{1}", e.X - rectangle.X, e.Y - rectangle.Y));
        }

        private void pbImg_MouseMove(object sender, MouseEventArgs e)
        {
            if (!isDown)
                return;
            //获取时间图片大小
            PropertyInfo rectangleProperty = this.pbImg.GetType().GetProperty("ImageRectangle", BindingFlags.Instance | BindingFlags.NonPublic);
            Rectangle rectangle = (Rectangle)rectangleProperty.GetValue(this.pbImg, null);
            DrawRectangle(e.X - rectangle.X, e.Y - rectangle.Y);
            DrawRectangleList();
        }

        private void pbImg_MouseUp(object sender, MouseEventArgs e)
        {
            isDown = false;
            //获取时间图片大小
            PropertyInfo rectangleProperty = this.pbImg.GetType().GetProperty("ImageRectangle", BindingFlags.Instance | BindingFlags.NonPublic);
            Rectangle rectangle = (Rectangle)rectangleProperty.GetValue(this.pbImg, null);
            DrawRectangle(e.X - rectangle.X, e.Y - rectangle.Y);
            DrawRectangleList();
            if (faceRectangle.Width > 0 && faceRectangle.Height > 0)
                pbSubImg.Image = new Bitmap(nowImg).Clone(faceReadRectangle, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            PrintMsg(String.Format("MouseUp:{0},{1}", e.X - rectangle.X, e.Y - rectangle.Y));
        }
        /// <summary>
        /// 计算矩形
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        private void DrawRectangle(int x, int y)
        {
            if (nowImg == null || btnStop.Enabled)
                return;
            //获取实际图片大小
            PropertyInfo rectangleProperty = this.pbImg.GetType().GetProperty("ImageRectangle", BindingFlags.Instance | BindingFlags.NonPublic);
            Rectangle rectangle = (Rectangle)rectangleProperty.GetValue(this.pbImg, null);
            int w = Math.Abs(x - faceRectangle.X), h = Math.Abs(y - faceRectangle.Y);
            if (faceRectangle.X > x)
            {
                faceRectangle.X = x;
            }
            if (faceRectangle.Y > y)
            {
                faceRectangle.Y = y;
            }
            faceRectangle.Width = w;
            faceRectangle.Height = h;
            double wb = rectangle.Width / (double)nowImg.Width;
            double hb = rectangle.Height / (double)nowImg.Height;
            faceReadRectangle = new Rectangle(faceRectangle.Location, faceRectangle.Size);
            faceReadRectangle.X = Convert.ToInt32(faceRectangle.X / wb);
            faceReadRectangle.Y = Convert.ToInt32(faceRectangle.Y / hb);
            faceReadRectangle.Width = Convert.ToInt32(faceRectangle.Width / wb);
            faceReadRectangle.Height = Convert.ToInt32(faceRectangle.Height / hb);

            //Bitmap cloneImg = new Bitmap(nowImg);
            //Graphics g = Graphics.FromImage(cloneImg);
            //int lineWide = Convert.ToInt32(2.0 / pbImg.Width * cloneImg.Width);
            //g.DrawRectangle(new Pen(Color.Red, lineWide), faceReadRectangle);
            //g.Dispose();
            //pbImg.Image = cloneImg;
            //GC.Collect();
        }
        /// <summary>
        /// 绘制矩形
        /// </summary>
        private void DrawRectangleList()
        {
            if (nowImg == null || btnStop.Enabled)
                return;
            Bitmap cloneImg = new Bitmap(nowImg);
            Graphics g = Graphics.FromImage(cloneImg);
            int lineWide = Convert.ToInt32(2.0 / pbImg.Width * cloneImg.Width);
            Pen pen = new Pen(Color.Red, lineWide);
            g.DrawRectangle(pen, faceReadRectangle);
            foreach (var r in faceReadRectangleList)
            {
                g.DrawRectangle(pen, r);
            }
            g.Dispose();
            pbImg.Image = cloneImg;
            GC.Collect();
        }

        private void PrintMsg(string msg)
        {
            txtMsg.Invoke(new Action(() =>
            {
                txtMsg.AppendText(msg + "\r\n");
            }));
        }
        private void GetFrame()
        {
            try
            {
                if (cap != null && cap.Ptr != IntPtr.Zero)
                {
                    if (!cap.IsOpened)
                        return;
                    cap.Retrieve(nextFrame, 0);
                    if (!nextFrame.IsEmpty)
                    {
                        pbImg.Image = nextFrame.Bitmap;
                    }
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.ToString());
            }
        }

        private void btnPredict_Click(object sender, EventArgs e)
        {
            Rectangle[] rects = PredictBP(nowImg);
            Bitmap cloneImg = new Bitmap(nowImg);
            Graphics g = Graphics.FromImage(cloneImg);
            int lineWide = Convert.ToInt32(2.0 / pbImg.Width * cloneImg.Width);
            foreach (var r in rects)
            {
                var tmpR = r;
                tmpR.X = (int)(tmpR.X / (float)bpWidth * cloneImg.Width);
                tmpR.Y = (int)(tmpR.Y / (float)bpHeight * cloneImg.Height);
                tmpR.Width = (int)(tmpR.Width / (float)bpWidth * cloneImg.Width);
                tmpR.Height = (int)(tmpR.Height / (float)bpHeight * cloneImg.Height);
                g.DrawRectangle(new Pen(Color.Red, lineWide), tmpR);
            }
            g.Dispose();
            pbImg.Image = cloneImg;
            MessageBox.Show("识别完成");
        }
        #region CNN神经网络
        //private int bpWidth = 960 / 2, bpHeight = 540 / 2;
        private int bpWidth = 1920, bpHeight = 1080;
        private int bpRectangleCount = 50;//人脸框最大数量
        private int bpTrainDataCount;//训练样本数
        private Cnn cnn;
        private void CreateBP()
        {
            cnn = new Cnn();
            cnn.AddCnnConvolutionLayer(8, bpWidth, bpHeight, 20, 20, 5, 5, 1, 2, 2, 2);
            cnn.AddCnnConvolutionLayer(20, 10, 10, 3, 3, 1, 2, 2, 2);
            cnn.AddCnnConvolutionLayer(40, 5, 5, 1, 1, 1, 2, 2, 2);
            cnn.AddCnnConvolutionLayer(60, 5, 5, 1, 1, 1, 2, 2, 2);
            //cnn.AddCnnConvolutionLayer(80, 5, 5, 1, 1, 1, 2, 2, 1);
            //cnn.AddCnnConvolutionLayer(100, 5, 5, 1, 1, 1, 2, 2, 1);
            //cnn.AddCnnFullLayer(300, 1);
            cnn.AddCnnFullLayer(bpRectangleCount * 4, 1);
        }
        private void TrainBP(Dictionary<Bitmap, List<Rectangle>> imgs)
        {
            bpTrainDataCount = imgs.Count;
            for (int tc = 0; tc < 10; tc++)
            {
                foreach (var item in imgs)
                {
                    Bitmap img = item.Key;
                    Rectangle[] rects = item.Value.ToArray();
                    //图片
                    Bitmap tmpImg = ZoomImg(img, bpWidth, bpHeight, ref rects);
                    Image<Bgr, float> trainingData = new Image<Bgr, float>(tmpImg);
                    double[,] input = new double[bpWidth, bpHeight];
                    for (int i = 0; i < bpWidth; i++)
                    {
                        for (int j = 0; j < bpHeight; j++)
                        {
                            input[i, j] = Color.FromArgb(
                                (int)trainingData.Data[j, i, 2],
                                (int)trainingData.Data[j, i, 1],
                                (int)trainingData.Data[j, i, 0]
                                ).ToArgb() / (float)0xFFFFFF;
                        }
                    }
                    //矩形数据
                    double[] output = new double[bpRectangleCount * 4];
                    for (int i = 0; i < rects.Length * 4 && i < bpRectangleCount * 4; i += 4)
                    {
                        output[i] = rects[i / 4].X / (float)bpWidth;
                        output[i + 1] = rects[i / 4].Y / (float)bpHeight;
                        output[i + 2] = rects[i / 4].Width / (float)bpWidth;
                        output[i + 3] = rects[i / 4].Height / (float)bpHeight;
                    }
                    tmpImg.Dispose();
                    tmpImg = null;
                    cnn.Train(input, output, 0.01);
                }
            }
        }
        private Rectangle[] PredictBP(Bitmap img)
        {
            Rectangle[] tmpR = new Rectangle[] { };
            Bitmap tmpImg = ZoomImg(img, bpWidth, bpHeight, ref tmpR);
            List<Rectangle> retRects = new List<Rectangle>();
            Image<Bgr, float> trainingData = new Image<Bgr, float>(tmpImg);
            double[,] input = new double[bpWidth, bpHeight];
            for (int i = 0; i < bpWidth; i++)
            {
                for (int j = 0; j < bpHeight; j++)
                {
                    input[i, j] = Color.FromArgb(
                        (int)trainingData.Data[j, i, 2],
                        (int)trainingData.Data[j, i, 1],
                        (int)trainingData.Data[j, i, 0]
                        ).ToArgb() / (float)0xFFFFFF;
                }
            }
            double[] output = cnn.Predict(input);
            for (int i = 0; i < bpRectangleCount * 4; i += 4)
            {
                Rectangle r = new Rectangle();
                r.X = (int)(output[i] * (float)bpWidth);
                r.Y = (int)(output[i + 1] * (float)bpHeight);
                r.Width = (int)(output[i + 2] * (float)bpWidth);
                r.Height = (int)(output[i + 3] * (float)bpHeight);
                if (r.X >= 0 && r.Y >= 0 && r.Width > 0 && r.Height > 0)
                    retRects.Add(r);
            }
            return retRects.ToArray();
        }

        private Bitmap ZoomImg(Bitmap img, int width, int height, ref Rectangle[] rects)
        {
            //缩放图片
            Bitmap tmpZoomImg = new Bitmap(width, height);
            Graphics g = Graphics.FromImage(tmpZoomImg);
            g.Clear(Color.Black);
            int x = 0, y = 0, w = width, h = height;
            double b1, b2;
            b1 = width / height;
            b2 = img.Width / img.Height;
            if (b1 > b2)
            {
                w = (int)(h * b2);
                x = (width - w) / 2;
            }
            else if (b2 > b1)
            {
                h = (int)(w / b2);
                y = (height - h) / 2;
            }
            g.DrawImage(img, x, y, w, h);
            g.Dispose();
            //转换矩形比例
            for (int i = 0; i < rects.Length; i++)
            {
                Rectangle rectangle = new Rectangle();
                double wb = img.Width / (double)width;
                double hb = img.Height / (double)height;
                rectangle.X = Convert.ToInt32(rects[i].X / wb);
                rectangle.Y = Convert.ToInt32(rects[i].Y / hb);
                rectangle.Width = Convert.ToInt32(rects[i].Width / wb);
                rectangle.Height = Convert.ToInt32(rects[i].Height / hb);
                rects[i] = rectangle;
            }
            return tmpZoomImg;
        }
        #endregion

        private void btnImport_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "*.jpg|*.jpg";
            if (ofd.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                using (FileStream fs = new FileStream(ofd.FileName, FileMode.Open))
                {
                    int byteLength = (int)fs.Length;
                    byte[] fileBytes = new byte[byteLength];
                    fs.Read(fileBytes, 0, byteLength);

                    nowImg = (Bitmap)Image.FromStream(new MemoryStream(fileBytes));
                    fs.Close();
                }
                pbImg.Image = nowImg;
                btnDeleteRectangle_Click(btnDeleteRectangle, null);
                if (File.Exists(ofd.FileName + ".txt"))
                {
                    using (StreamReader sr = new StreamReader(ofd.FileName + ".txt"))
                    {
                        string tmpStrRectangle = sr.ReadLine();
                        while (!String.IsNullOrEmpty(tmpStrRectangle))
                        {
                            string[] tmpStrRectangleItem = tmpStrRectangle.Split(' ');
                            Rectangle tmpRectangle = new Rectangle();
                            tmpRectangle.X = Convert.ToInt32(tmpStrRectangleItem[0]);
                            tmpRectangle.Y = Convert.ToInt32(tmpStrRectangleItem[1]);
                            tmpRectangle.Width = Convert.ToInt32(tmpStrRectangleItem[2]);
                            tmpRectangle.Height = Convert.ToInt32(tmpStrRectangleItem[3]);
                            faceReadRectangleList.Add(tmpRectangle);
                            tmpStrRectangle = sr.ReadLine();
                        }
                        sr.Close();
                    }

                }
                lblRectangleCount.Text = faceReadRectangleList.Count.ToString();
                DrawRectangleList();
                MessageBox.Show("导入完成");
            }
        }

        private void btnExport_Click(object sender, EventArgs e)
        {
            if (nowImg == null || btnStop.Enabled)
                return;
            SaveFileDialog sfd = new SaveFileDialog();
            sfd.Filter = "*.jpg|*.jpg";
            if (sfd.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                nowImg.Save(sfd.FileName, System.Drawing.Imaging.ImageFormat.Jpeg);
                using (StreamWriter fs = new StreamWriter(sfd.FileName + ".txt", false))
                {
                    foreach (var r in faceReadRectangleList)
                    {
                        fs.WriteLine(String.Format("{0} {1} {2} {3}", r.X, r.Y, r.Width, r.Height));
                    }
                    fs.Close();
                }
                MessageBox.Show("导出完成");
            }
        }

        private void btn_Click(object sender, EventArgs e)
        {
            FolderBrowserDialog fbd = new FolderBrowserDialog();
            if (fbd.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                Dictionary<Bitmap, List<Rectangle>> nowImgs = new Dictionary<Bitmap, List<Rectangle>>();
                foreach (var path in Directory.EnumerateFiles(fbd.SelectedPath, "*.jpg"))
                {
                    Bitmap tmpImg;
                    using (FileStream fs = new FileStream(path, FileMode.Open))
                    {
                        int byteLength = (int)fs.Length;
                        byte[] fileBytes = new byte[byteLength];
                        fs.Read(fileBytes, 0, byteLength);

                        tmpImg = (Bitmap)Image.FromStream(new MemoryStream(fileBytes));
                        fs.Close();
                    }
                    List<Rectangle> tmpListRectangle = new List<Rectangle>();
                    if (File.Exists(path + ".txt"))
                    {
                        using (StreamReader sr = new StreamReader(path + ".txt"))
                        {
                            string tmpStrRectangle = sr.ReadLine();
                            while (!String.IsNullOrEmpty(tmpStrRectangle))
                            {
                                string[] tmpStrRectangleItem = tmpStrRectangle.Split(' ');
                                Rectangle tmpRectangle = new Rectangle();
                                tmpRectangle.X = Convert.ToInt32(tmpStrRectangleItem[0]);
                                tmpRectangle.Y = Convert.ToInt32(tmpStrRectangleItem[1]);
                                tmpRectangle.Width = Convert.ToInt32(tmpStrRectangleItem[2]);
                                tmpRectangle.Height = Convert.ToInt32(tmpStrRectangleItem[3]);
                                tmpListRectangle.Add(tmpRectangle);
                                tmpStrRectangle = sr.ReadLine();
                            }
                            sr.Close();
                        }

                    }
                    nowImgs.Add(tmpImg, tmpListRectangle);
                }
                TrainBP(nowImgs);
                MessageBox.Show("训练完成");
            }
        }

        private void btnSaveBP_Click(object sender, EventArgs e)
        {
            SaveFileDialog sfd = new SaveFileDialog();
            sfd.Filter = "XML文件(*.xml)|*.xml";
            if (sfd.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                //cnn.Write(new FileStorage(sfd.FileName, FileStorage.Mode.Write));
                MessageBox.Show("导出完成");
            }
        }

        private void btnLoadBP_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "XML文件(*.xml)|*.xml";
            if (ofd.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                //cnn.Read(new FileStorage(ofd.FileName, FileStorage.Mode.Read).GetRoot());
                MessageBox.Show("导入完成");
            }
        }

        private void btnAddRectangle_Click(object sender, EventArgs e)
        {
            faceReadRectangleList.Add(new Rectangle(faceReadRectangle.Location, faceReadRectangle.Size));
            faceReadRectangle.Height = 0;
            faceReadRectangle.Width = 0;
            lblRectangleCount.Text = faceReadRectangleList.Count.ToString();
            DrawRectangleList();
        }

        private void btnDeleteRectangle_Click(object sender, EventArgs e)
        {
            faceReadRectangleList.Clear();
            faceReadRectangle.Location = Point.Empty;
            faceReadRectangle.Size = Size.Empty;
            lblRectangleCount.Text = faceReadRectangleList.Count.ToString();
            DrawRectangleList();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            //if (args.length < 5)
            //{
            //    System.out
            //            .println("Usage: \n\t-train trainfile\n\t-test predictfile\n\t-sep separator, default:','\n\t-eta eta, default:0.5\n\t-iter iternum, default:5000\n\t-out outputfile");
            //    return;
            //}
            String trainfile = @".\data\train.txt";
            String testfile = @".\data\test.txt";
            String outputfile = "outputfile.txt";
            float eta = 0.01f;
            int nIter = 1000;
            List<EmguCVDemo.BP.DataNode> trainList = GetDataList(trainfile);
            List<EmguCVDemo.BP.DataNode> testList = GetDataList(testfile);
            StreamWriter sw = new StreamWriter(outputfile);
            int typeCount = 3;
            Cnn tmpCnn = new Cnn();
            tmpCnn.AddCnnFullLayer(4, 14, 1);
            tmpCnn.AddCnnFullLayer(3, 1);
            for (int i = 0; i < nIter; i++)
            {
                foreach (var t in trainList)
                {
                    tmpCnn.TrainFullLayer(t.getAttribList2().ToArray(), t.getTypes(), eta);
                }
            }
            for (int i = 0; i < testList.Count(); i++)
            {
                EmguCVDemo.BP.DataNode test = testList[i];
                double[] type = tmpCnn.PredictFullLayer(test.getAttribList2().ToArray());
                List<float> attribs = test.getAttribList();
                for (int n = 0; n < attribs.Count(); n++)
                {
                    sw.Write(attribs[n] + ",");
                }
                double maxtype = type[0], max = 0;
                for (int n = 0; n < 3; n++)
                {
                    if (maxtype < type[n])
                    {
                        max = n;
                        maxtype = type[n];
                    }
                }
                sw.WriteLine(GetTypeName(max));
            }
            sw.Close();
        }
        private List<EmguCVDemo.BP.DataNode> GetDataList(string path)
        {
            List<EmguCVDemo.BP.DataNode> tmpListDataNode = new List<BP.DataNode>();
            StreamReader sr = new StreamReader(path);
            string line = sr.ReadLine();
            while (!String.IsNullOrEmpty(line))
            {
                string[] values = line.Split(',');
                EmguCVDemo.BP.DataNode data = new BP.DataNode();
                for (int i = 0; i < values.Length; i++)
                {
                    if (i < values.Length - 1)
                    {
                        data.addAttrib(Convert.ToSingle(values[i]));
                    }
                    else
                    {
                        switch (values[i])
                        {
                            case "Iris-versicolor":
                                data.setType(0);
                                break;
                            case "Iris-setosa":
                                data.setType(1);
                                break;
                            case "Iris-virginica":
                                data.setType(2);
                                break;
                        }
                    }
                }
                tmpListDataNode.Add(data);
                line = sr.ReadLine();
            }
            sr.Close();
            return tmpListDataNode;
        }

        private string GetTypeName(double type)
        {
            switch ((int)Math.Round(type, 0))
            {
                case 0:
                    return "Iris-versicolor";
                case 1:
                    return "Iris-setosa";
                case 2:
                    return "Iris-virginica";
            }
            return "";
        }
    }
}

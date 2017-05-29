namespace EmguCVDemo
{
    partial class Form1
    {
        /// <summary>
        /// 必需的设计器变量。
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// 清理所有正在使用的资源。
        /// </summary>
        /// <param name="disposing">如果应释放托管资源，为 true；否则为 false。</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows 窗体设计器生成的代码

        /// <summary>
        /// 设计器支持所需的方法 - 不要
        /// 使用代码编辑器修改此方法的内容。
        /// </summary>
        private void InitializeComponent()
        {
            this.pbImg = new System.Windows.Forms.PictureBox();
            this.panel1 = new System.Windows.Forms.Panel();
            this.pbSubImg = new System.Windows.Forms.PictureBox();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.btnTrain = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.lblRectangleCount = new System.Windows.Forms.Label();
            this.btnDeleteRectangle = new System.Windows.Forms.Button();
            this.btnAddRectangle = new System.Windows.Forms.Button();
            this.btnPredict = new System.Windows.Forms.Button();
            this.btnSaveBP = new System.Windows.Forms.Button();
            this.btnLoadBP = new System.Windows.Forms.Button();
            this.btnExport = new System.Windows.Forms.Button();
            this.btnImport = new System.Windows.Forms.Button();
            this.button2 = new System.Windows.Forms.Button();
            this.btn = new System.Windows.Forms.Button();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.btnStart = new System.Windows.Forms.Button();
            this.btnPause = new System.Windows.Forms.Button();
            this.btnStop = new System.Windows.Forms.Button();
            this.txtMsg = new System.Windows.Forms.TextBox();
            ((System.ComponentModel.ISupportInitialize)(this.pbImg)).BeginInit();
            this.panel1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pbSubImg)).BeginInit();
            this.groupBox1.SuspendLayout();
            this.groupBox2.SuspendLayout();
            this.SuspendLayout();
            // 
            // pbImg
            // 
            this.pbImg.Dock = System.Windows.Forms.DockStyle.Fill;
            this.pbImg.Location = new System.Drawing.Point(200, 0);
            this.pbImg.Name = "pbImg";
            this.pbImg.Size = new System.Drawing.Size(605, 549);
            this.pbImg.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pbImg.TabIndex = 0;
            this.pbImg.TabStop = false;
            this.pbImg.MouseDown += new System.Windows.Forms.MouseEventHandler(this.pbImg_MouseDown);
            this.pbImg.MouseMove += new System.Windows.Forms.MouseEventHandler(this.pbImg_MouseMove);
            this.pbImg.MouseUp += new System.Windows.Forms.MouseEventHandler(this.pbImg_MouseUp);
            // 
            // panel1
            // 
            this.panel1.Controls.Add(this.pbSubImg);
            this.panel1.Controls.Add(this.groupBox1);
            this.panel1.Controls.Add(this.groupBox2);
            this.panel1.Controls.Add(this.txtMsg);
            this.panel1.Dock = System.Windows.Forms.DockStyle.Left;
            this.panel1.Location = new System.Drawing.Point(0, 0);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(200, 549);
            this.panel1.TabIndex = 1;
            // 
            // pbSubImg
            // 
            this.pbSubImg.Dock = System.Windows.Forms.DockStyle.Fill;
            this.pbSubImg.Location = new System.Drawing.Point(0, 257);
            this.pbSubImg.Name = "pbSubImg";
            this.pbSubImg.Size = new System.Drawing.Size(200, 209);
            this.pbSubImg.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pbSubImg.TabIndex = 4;
            this.pbSubImg.TabStop = false;
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.btnTrain);
            this.groupBox1.Controls.Add(this.label1);
            this.groupBox1.Controls.Add(this.lblRectangleCount);
            this.groupBox1.Controls.Add(this.btnDeleteRectangle);
            this.groupBox1.Controls.Add(this.btnAddRectangle);
            this.groupBox1.Controls.Add(this.btnPredict);
            this.groupBox1.Controls.Add(this.btnSaveBP);
            this.groupBox1.Controls.Add(this.btnLoadBP);
            this.groupBox1.Controls.Add(this.btnExport);
            this.groupBox1.Controls.Add(this.btnImport);
            this.groupBox1.Controls.Add(this.button2);
            this.groupBox1.Controls.Add(this.btn);
            this.groupBox1.Dock = System.Windows.Forms.DockStyle.Top;
            this.groupBox1.Location = new System.Drawing.Point(0, 55);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(200, 202);
            this.groupBox1.TabIndex = 2;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "训练";
            // 
            // btnTrain
            // 
            this.btnTrain.Location = new System.Drawing.Point(9, 107);
            this.btnTrain.Name = "btnTrain";
            this.btnTrain.Size = new System.Drawing.Size(75, 23);
            this.btnTrain.TabIndex = 11;
            this.btnTrain.Text = "训练";
            this.btnTrain.UseVisualStyleBackColor = true;
            this.btnTrain.Visible = false;
            this.btnTrain.Click += new System.EventHandler(this.btnTrain_Click);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(109, 173);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(59, 12);
            this.label1.TabIndex = 10;
            this.label1.Text = "样本数：0";
            // 
            // lblRectangleCount
            // 
            this.lblRectangleCount.AutoSize = true;
            this.lblRectangleCount.Location = new System.Drawing.Point(12, 173);
            this.lblRectangleCount.Name = "lblRectangleCount";
            this.lblRectangleCount.Size = new System.Drawing.Size(59, 12);
            this.lblRectangleCount.TabIndex = 9;
            this.lblRectangleCount.Text = "人脸框：0";
            // 
            // btnDeleteRectangle
            // 
            this.btnDeleteRectangle.Location = new System.Drawing.Point(111, 136);
            this.btnDeleteRectangle.Name = "btnDeleteRectangle";
            this.btnDeleteRectangle.Size = new System.Drawing.Size(75, 23);
            this.btnDeleteRectangle.TabIndex = 8;
            this.btnDeleteRectangle.Text = "移除框";
            this.btnDeleteRectangle.UseVisualStyleBackColor = true;
            this.btnDeleteRectangle.Click += new System.EventHandler(this.btnDeleteRectangle_Click);
            // 
            // btnAddRectangle
            // 
            this.btnAddRectangle.Location = new System.Drawing.Point(9, 136);
            this.btnAddRectangle.Name = "btnAddRectangle";
            this.btnAddRectangle.Size = new System.Drawing.Size(75, 23);
            this.btnAddRectangle.TabIndex = 7;
            this.btnAddRectangle.Text = "保存框";
            this.btnAddRectangle.UseVisualStyleBackColor = true;
            this.btnAddRectangle.Click += new System.EventHandler(this.btnAddRectangle_Click);
            // 
            // btnPredict
            // 
            this.btnPredict.Location = new System.Drawing.Point(111, 107);
            this.btnPredict.Name = "btnPredict";
            this.btnPredict.Size = new System.Drawing.Size(75, 23);
            this.btnPredict.TabIndex = 6;
            this.btnPredict.Text = "识别";
            this.btnPredict.UseVisualStyleBackColor = true;
            this.btnPredict.Click += new System.EventHandler(this.btnPredict_Click);
            // 
            // btnSaveBP
            // 
            this.btnSaveBP.Location = new System.Drawing.Point(111, 78);
            this.btnSaveBP.Name = "btnSaveBP";
            this.btnSaveBP.Size = new System.Drawing.Size(75, 23);
            this.btnSaveBP.TabIndex = 5;
            this.btnSaveBP.Text = "导出网络";
            this.btnSaveBP.UseVisualStyleBackColor = true;
            this.btnSaveBP.Click += new System.EventHandler(this.btnSaveBP_Click);
            // 
            // btnLoadBP
            // 
            this.btnLoadBP.Location = new System.Drawing.Point(9, 78);
            this.btnLoadBP.Name = "btnLoadBP";
            this.btnLoadBP.Size = new System.Drawing.Size(75, 23);
            this.btnLoadBP.TabIndex = 4;
            this.btnLoadBP.Text = "导入网络";
            this.btnLoadBP.UseVisualStyleBackColor = true;
            this.btnLoadBP.Click += new System.EventHandler(this.btnLoadBP_Click);
            // 
            // btnExport
            // 
            this.btnExport.Location = new System.Drawing.Point(111, 20);
            this.btnExport.Name = "btnExport";
            this.btnExport.Size = new System.Drawing.Size(75, 23);
            this.btnExport.TabIndex = 3;
            this.btnExport.Text = "导出图片";
            this.btnExport.UseVisualStyleBackColor = true;
            this.btnExport.Click += new System.EventHandler(this.btnExport_Click);
            // 
            // btnImport
            // 
            this.btnImport.Location = new System.Drawing.Point(9, 20);
            this.btnImport.Name = "btnImport";
            this.btnImport.Size = new System.Drawing.Size(75, 23);
            this.btnImport.TabIndex = 2;
            this.btnImport.Text = "导入图片";
            this.btnImport.UseVisualStyleBackColor = true;
            this.btnImport.Click += new System.EventHandler(this.btnImport_Click);
            // 
            // button2
            // 
            this.button2.Location = new System.Drawing.Point(111, 49);
            this.button2.Name = "button2";
            this.button2.Size = new System.Drawing.Size(75, 23);
            this.button2.TabIndex = 1;
            this.button2.Text = "增加负样本";
            this.button2.UseVisualStyleBackColor = true;
            // 
            // btn
            // 
            this.btn.Location = new System.Drawing.Point(9, 49);
            this.btn.Name = "btn";
            this.btn.Size = new System.Drawing.Size(75, 23);
            this.btn.TabIndex = 0;
            this.btn.Text = "增加正样本";
            this.btn.UseVisualStyleBackColor = true;
            this.btn.Click += new System.EventHandler(this.btn_Click);
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.btnStart);
            this.groupBox2.Controls.Add(this.btnPause);
            this.groupBox2.Controls.Add(this.btnStop);
            this.groupBox2.Dock = System.Windows.Forms.DockStyle.Top;
            this.groupBox2.Location = new System.Drawing.Point(0, 0);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(200, 55);
            this.groupBox2.TabIndex = 5;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "摄像头";
            // 
            // btnStart
            // 
            this.btnStart.Location = new System.Drawing.Point(12, 20);
            this.btnStart.Name = "btnStart";
            this.btnStart.Size = new System.Drawing.Size(55, 23);
            this.btnStart.TabIndex = 0;
            this.btnStart.Text = "启动";
            this.btnStart.UseVisualStyleBackColor = true;
            this.btnStart.Click += new System.EventHandler(this.btnStart_Click);
            // 
            // btnPause
            // 
            this.btnPause.Location = new System.Drawing.Point(73, 20);
            this.btnPause.Name = "btnPause";
            this.btnPause.Size = new System.Drawing.Size(55, 23);
            this.btnPause.TabIndex = 1;
            this.btnPause.Text = "暂停";
            this.btnPause.UseVisualStyleBackColor = true;
            this.btnPause.Visible = false;
            this.btnPause.Click += new System.EventHandler(this.btnPause_Click);
            // 
            // btnStop
            // 
            this.btnStop.Location = new System.Drawing.Point(134, 20);
            this.btnStop.Name = "btnStop";
            this.btnStop.Size = new System.Drawing.Size(55, 23);
            this.btnStop.TabIndex = 2;
            this.btnStop.Text = "停止";
            this.btnStop.UseVisualStyleBackColor = true;
            this.btnStop.Click += new System.EventHandler(this.btnStop_Click);
            // 
            // txtMsg
            // 
            this.txtMsg.Dock = System.Windows.Forms.DockStyle.Bottom;
            this.txtMsg.Location = new System.Drawing.Point(0, 466);
            this.txtMsg.Multiline = true;
            this.txtMsg.Name = "txtMsg";
            this.txtMsg.ReadOnly = true;
            this.txtMsg.Size = new System.Drawing.Size(200, 83);
            this.txtMsg.TabIndex = 3;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 12F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(805, 549);
            this.Controls.Add(this.pbImg);
            this.Controls.Add(this.panel1);
            this.Name = "Form1";
            this.Text = "Form1";
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.pbImg)).EndInit();
            this.panel1.ResumeLayout(false);
            this.panel1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pbSubImg)).EndInit();
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.groupBox2.ResumeLayout(false);
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.PictureBox pbImg;
        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.Button btnStop;
        private System.Windows.Forms.Button btnPause;
        private System.Windows.Forms.Button btnStart;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.Button button2;
        private System.Windows.Forms.Button btn;
        private System.Windows.Forms.TextBox txtMsg;
        private System.Windows.Forms.PictureBox pbSubImg;
        private System.Windows.Forms.Button btnSaveBP;
        private System.Windows.Forms.Button btnLoadBP;
        private System.Windows.Forms.Button btnExport;
        private System.Windows.Forms.Button btnImport;
        private System.Windows.Forms.Button btnPredict;
        private System.Windows.Forms.Button btnDeleteRectangle;
        private System.Windows.Forms.Button btnAddRectangle;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.Label lblRectangleCount;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Button btnTrain;
    }
}


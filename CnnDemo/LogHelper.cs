using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using log4net;
using System.Reflection;
using log4net.Config;

namespace CnnDemo
{
    public static class LogHelper
    {
        public static ILog m_log;
        public static void Init()
        {
            if (m_log == null)
            {
                XmlConfigurator.Configure();
                Type type = MethodBase.GetCurrentMethod().DeclaringType;
                m_log = LogManager.GetLogger(type);
            }
        }
        public static void Info(string msg)
        {
            Init();
            //m_log.Debug("这是一个Debug日志");
            m_log.Info(msg);
            //m_log.Warn("这是一个Warn日志");
            //m_log.Error("这是一个Error日志");
            //m_log.Fatal("这是一个Fatal日志");
        }
    }
}

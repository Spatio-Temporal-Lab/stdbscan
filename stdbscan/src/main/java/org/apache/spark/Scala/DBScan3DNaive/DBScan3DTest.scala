package org.apache.spark.Scala.DBScan3DNaive


import org.apache.spark.Scala.utils.file.FileProcess
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.{SparkConf, SparkContext}


import java.text.SimpleDateFormat
import java.util.Date

object DBScan3DTest {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setAppName("DBscan_3D")
      .setMaster("spark://10.242.6.19:7077")
    val sparkContext: SparkContext = new SparkContext(conf)

    val fileProcess: FileProcess = FileProcess()



    val fileList = Array[String](args(0))

    val lineRDD: RDD[String] = sparkContext.textFile(fileList.mkString(","), 5)
    val VectorRDD: RDD[Vector] = lineRDD.map((x: String) => {
      
      fileProcess.NewYorkDataProcess(x)
    }).map((x: (Double, Double, Double)) => {
      Vectors.dense(Array(x._1, x._2, x._3))
    })

    val distanceEps: Double = args(2).toDouble / 10

    val timeEps: Double = args(3).toDouble
    val minPoints: Int = args(4).toInt
    val numberOfPartitions: Int = args(5).toInt
    // new param: load_balance_alpha
    val load_balance_alpha = args(6).toDouble
    // new partition method params
    val x_boundind: Double = args(7).toDouble
    val y_bounding: Double = args(8).toDouble
    val t_bounding: Double = args(9).toDouble
    val startTime = System.currentTimeMillis()
  
    val DBScanRes = DBScan3D_CubeSplit.train(VectorRDD, distanceEps, timeEps, minPoints, numberOfPartitions, load_balance_alpha, x_boundind, y_bounding, t_bounding)

    val endTime = System.currentTimeMillis()
    val total = endTime - startTime
    println("-----------------------------------------------------------------")
    println(s"Total Time Cost: $total")
    sparkContext.stop()
  }
}
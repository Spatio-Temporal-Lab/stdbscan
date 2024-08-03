package org.apache.spark.Scala.utils.sample

import org.apache.spark.Scala.DBScan3DNaive.DBScanPoint_3D
import org.apache.spark.rdd.RDD
import org.apache.spark.util.random.SamplingUtils
import org.apache.spark.mllib.linalg.Vector
object Sample {
  def sample(rdd: RDD[Vector], sampleRate: Double): RDD[DBScanPoint_3D] = {
    val samples = rdd.sample(withReplacement = false, sampleRate, seed = 9961).map((x) => DBScanPoint_3D(x))
    samples
  }

  def strict_sample(rdd: RDD[Vector], count: Int): Array[DBScanPoint_3D] = {
    val samples = rdd.takeSample(withReplacement = false, num = count, seed = 9961).map(x => DBScanPoint_3D(x))
    samples
  }
}

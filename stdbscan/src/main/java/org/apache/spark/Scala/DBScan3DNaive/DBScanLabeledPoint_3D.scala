package org.apache.spark.Scala.DBScan3DNaive

import org.apache.spark.mllib.linalg.Vector
object DBScanLabeledPoint_3D {
  val Unknown = 0

  object Flag extends Enumeration{ 
    type Flag = Value
    val Border, Core, Noise, NotFlagged = Value
  }
} 


class DBScanLabeledPoint_3D(vector: Vector) extends DBScanPoint_3D(vector){
  require(vector != null, "Vector should not be null")
  def this(point: DBScanPoint_3D) = this(point.vector)

  var flag = DBScanLabeledPoint_3D.Flag.NotFlagged
  var cluster = DBScanLabeledPoint_3D.Unknown
  var visited = false

  override def toString: String = {
    s"$vector, $cluster, $flag"
  }
}

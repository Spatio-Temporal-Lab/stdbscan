package org.apache.spark.Scala.DBScan3DNaive

import org.apache.spark.mllib.linalg.Vector
object DBScanPoint_3D{
  val EARTH_R = 6378.137
}
case class DBScanPoint_3D(vector: Vector) {
  require(vector != null && vector.size == 3, "Vector cannot be null")
  def distanceX: Double = {

    vector(0)
  }
  def distanceY: Double = {

    vector(1)
  }
  def timeDimension: Double = {

    vector(2)
  }
  def distanceSquared(other: DBScanPoint_3D): Double = {
    require(other != null, "Other DBScanPoint_3D cannot be null")
    require(other.vector != null, "Other DBScanPoint_3D.vector cannot be null")
    val dx = other.distanceX - distanceX
    val dy = other.distanceY - distanceY
    dx * dx + dy * dy
  }
  def timeDistance(other: DBScanPoint_3D): Double = {
    require(other != null, "Other DBScanPoint_3D cannot be null")
    require(other.vector != null, "Other DBScanPoint_3D's vector cannot be null")

    val tx = other.timeDimension - timeDimension
    tx
  }
}

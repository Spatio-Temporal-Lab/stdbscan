package org.apache.spark.Scala.DBScan3DNaive

/*
* A cuboid with left down corner of (x, y, t) and right upper corner of (x2, y2 ,t2)
* */
case class DBScanCube(x: Double, y: Double, t: Double, x2: Double, y2: Double, t2: Double){
  /**
   * return another cube is contained by this one or not
   * @param other
   * @return
   */
  def contains(other: DBScanCube): Boolean = {
    x <= other.x && other.x2 <= x2 && y <= other.y && other.y2 <= y2 && t <= other.t && other.t2 <= t2
  }

  /**
   * return the three dimension is contained by this cube or not
   * @param point
   * @return
   */
  def contains(point: DBScanPoint_3D): Boolean = {
    x <= point.distanceX && point.distanceX <= x2 && y <= point.distanceY && point.distanceY <= y2 && t <= point.timeDimension && point.timeDimension <= t2
  }

  /**
   * Return the new DBSCANCuboid from shrinking this rectangle by given amount
   * if the mount is positive, the rectangle becomes the inner rectangle, otherwise, it becomes the outer rectangle
   * @param amount
   * @param amount2
   * @return
   */
  def shrink(amount: Double,amount2: Double): DBScanCube = {
    DBScanCube(x + amount, y + amount, t + amount2, x2 - amount, y2 - amount, t2 - amount2)
  }

  /**
   * return whther the point which is three dimension is almost contained by this cube or not
   * @param point
   * @return
   */
  def almostContains(point: DBScanPoint_3D): Boolean = {
    x < point.distanceX && point.distanceX < x2 && y < point.distanceY && point.distanceY < y2 && t < point.timeDimension && point.timeDimension < t2
  }
}

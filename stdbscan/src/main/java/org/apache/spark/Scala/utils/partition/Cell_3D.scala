package org.apache.spark.Scala.utils.partition

import org.apache.spark.Scala.DBScan3DNaive.{DBScanCube, DBScanPoint_3D}


object Cell_3D{
  /**
   * Split the giving Space into some Cubes
   * @param x_bounding
   * @param y_bounding
   * @param t_bounding
   * @return the Set of Cubes
   */
  def getCube(points:Array[DBScanPoint_3D], x_bounding: Double, y_bounding: Double, t_bounding: Double): Set[(Int, DBScanCube, Int)] = {
    new Cell_3D(points, x_bounding, y_bounding, t_bounding).getSplits()
  }
}


case class Cell_3D(points:Array[DBScanPoint_3D], x_bounding: Double, y_bounding: Double, t_bounding: Double) {
  type CubeWithCount = (DBScanCube, Int)

  def pointsIn(cube: DBScanCube): Int = {
    var count = 0
    for (point <- points){
      if(point != null && cube.contains(point)){
        count = count + 1
      }
    }
    count

  }

  def getMinimalBounding(points: Array[DBScanPoint_3D]): DBScanCube = {
    val x_min: Double = points.map(x => x.distanceX).toList.min
    val x_max: Double = points.map(x => x.distanceX).toList.max
    val y_min: Double = points.map(x => x.distanceY).toList.min
    val y_max: Double = points.map(x => x.distanceY).toList.max
    val t_min: Double = points.map(x => x.timeDimension).toList.min
    val t_max: Double = points.map(x => x.timeDimension).toList.max
    DBScanCube(x_min, y_min, t_min, x_max, y_max, t_max)
  }
  def getSplits(): Set[(Int, DBScanCube, Int)] ={
    val boundingCube: DBScanCube = getMinimalBounding(points)
    doSplitWithBounding(boundingCube, x_bounding, y_bounding, t_bounding)
  }
  def doSplitWithBounding(originCube: DBScanCube, x_bounding: Double, y_bounding: Double, t_bounding: Double): Set[(Int, DBScanCube, Int)] = {
    val functionTimeBegin = System.currentTimeMillis()
    val x_split = (originCube.x until originCube.x2 by x_bounding).toList :+ originCube.x2
    val y_split = (originCube.y until originCube.y2 by y_bounding).toList :+ originCube.y2
    val t_split = (originCube.t until originCube.t2 by t_bounding).toList :+ originCube.t2

    val cubes = for {
      x <- x_split.init
      y <- y_split.init
      t <- t_split.init
    } yield DBScanCube(x, y, t, x + x_bounding, y + y_bounding, t + t_bounding)
    val cubeTemp = cubes.toSet

    val filteredCube = cubeTemp.map({
      cube => (cube, pointsIn(cube))
    })
      .filter({
        case (_, count) => count != 0
      })


    var index = 0
    val IndexCube = filteredCube.map({
      case (cube, count) => {
        index = index + 1
        (index, cube, count)
      }
    })
    val functionTimeEnd = System.currentTimeMillis()
    val cost = functionTimeBegin - functionTimeEnd
    IndexCube
  }
}

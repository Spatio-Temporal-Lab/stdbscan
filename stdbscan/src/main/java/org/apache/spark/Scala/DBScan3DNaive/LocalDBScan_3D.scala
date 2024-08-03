package org.apache.spark.Scala.DBScan3DNaive

import org.apache.spark.Scala.DBScan3DNaive.DBScanLabeledPoint_3D.Flag
import org.apache.spark.internal.Logging

import scala.collection.mutable

class LocalDBScan_3D(distanceEps: Double, timeEps: Double, minPoints: Int) extends Logging {
  val minDistanceSquared = distanceEps * distanceEps
  val minTimeAbs = timeEps

  def fit(points: Iterable[DBScanPoint_3D]): Iterable[DBScanLabeledPoint_3D] = {
    require(points != null, "Points should not be null")

    val labeledPoints = points.map{new DBScanLabeledPoint_3D(_)}.toArray

    val totalClusters = labeledPoints.foldLeft(DBScanLabeledPoint_3D.Unknown) { (cluster, point) =>
      if (point == null) {
        cluster
      } else {
        if (!point.visited) {
          point.visited = true
          val neighbors = findNeighbors(point, labeledPoints)
          if (neighbors.size < minPoints) {
            point.flag = Flag.Noise
            cluster
          } else {

            expandCluster(point, neighbors, labeledPoints, cluster + 1)
            cluster + 1
          }
        } else {
          cluster
        }
      }
    }

    labeledPoints
  }

  private def findNeighbors(point: DBScanPoint_3D, all: Array[DBScanLabeledPoint_3D]): Iterable[DBScanLabeledPoint_3D] = all.filter((other: DBScanLabeledPoint_3D) => {
      other != null && point.distanceSquared(other) <= minDistanceSquared && math.abs(point.timeDistance(other)) <= minTimeAbs
    })


  def expandCluster(point: DBScanLabeledPoint_3D,
                    neighbors: Iterable[DBScanLabeledPoint_3D],
                    all: Array[DBScanLabeledPoint_3D],
                    cluster: Int): Unit = {
    require(point != null, "Point cannot be null")
    require(neighbors != null, "Neighbors cannot be null")
    require(all != null, "All points array cannot be null")

      point.flag = Flag.Core
      point.cluster = cluster
      var allNeighbors = mutable.Queue(neighbors)
      while (allNeighbors.nonEmpty) {
        allNeighbors.dequeue().foreach(neighbor => {
          if (neighbor != null) {
            if (!neighbor.visited) {
              neighbor.visited = true
              neighbor.cluster = cluster
              val neighborNeighbors = findNeighbors(neighbor, all)
              if (neighborNeighbors != null && neighborNeighbors.size >= minPoints) {
                neighbor.flag = Flag.Core
                allNeighbors.enqueue(neighborNeighbors)
              } else {
                neighbor.flag = Flag.Border
              }
            }
            if (neighbor.cluster == DBScanLabeledPoint_3D.Unknown) {
              neighbor.cluster = cluster
              neighbor.flag = Flag.Border
            } 
          }
        })
      }

  }
}

package org.apache.spark.Scala.utils.partition

import org.apache.spark.Scala.DBScan3DNaive.DBScanCube

import scala.collection.mutable
import scala.math.sqrt

object CellGraph_3D{

  def getCellGraph(pointOfCube:Set[(Int, DBScanCube, Int)],x_bounding: Double,y_bounding: Double,t_bounding: Double): Graph = {
    new CellGraph_3D(pointOfCube,x_bounding,y_bounding,t_bounding).getGraph()
  }

}


case class CellGraph_3D(pointOfCube:Set[(Int, DBScanCube, Int)],x_bounding: Double,y_bounding: Double,t_bounding: Double) {
  def neighbor(cube1: DBScanCube, cube2: DBScanCube): Boolean ={
    val dx = Math.abs(cube1.x - cube2.x)
    val dy = Math.abs(cube1.y - cube2.y)
    val dt = Math.abs(cube1.t - cube2.t)
    dx <= x_bounding && dy <= y_bounding && dt <= t_bounding
  }


  def getGraph(): Graph={
    val functionTimeBegin = System.currentTimeMillis()
    var vertices: mutable.SortedSet[Int] = mutable.SortedSet()
    var edges: mutable.Map[(Int, Int), Double] = mutable.Map()

    var neighborsMap: Map[Int, Set[Int]] = Map()
    for ((id1, cube1, count1) <- pointOfCube) {
      for ((id2, cube2, count2) <- pointOfCube if id1 < id2) {
        if(neighbor(cube1,cube2)){

          val weight = sqrt(count1 * count2)

          vertices += id1
          vertices += id2

          edges += (id1, id2) -> weight
          edges += (id2, id1) -> weight
          neighborsMap += id1 -> (neighborsMap.getOrElse(id1, Set()) + id2)
          neighborsMap += id2 -> (neighborsMap.getOrElse(id2, Set()) + id1) 
        }
      }
    }
    for ((id,_,_) <- pointOfCube if !neighborsMap.contains(id)) {
      vertices += id
      edges += (id, -1) -> 0
    }

    val immutableEdges = edges.toMap
    val functionTimeEnd = System.currentTimeMillis()
    val cost = functionTimeBegin - functionTimeEnd
 

    Graph(vertices, immutableEdges)
  }
}
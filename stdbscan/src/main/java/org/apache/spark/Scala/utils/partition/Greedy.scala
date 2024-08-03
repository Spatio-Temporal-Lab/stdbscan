package org.apache.spark.Scala.utils.partition
import org.apache.spark.Scala.DBScan3DNaive.DBScanCube

import scala.collection.mutable

object Greedy{
  def getGreedyPartition(pointOfCube:Set[(Int, DBScanCube, Int)], cellGraph:Graph, PointsPerPartition:Int,kk:Int): List[Set[DBScanCube]] = {
    new Greedy(pointOfCube,cellGraph,PointsPerPartition,kk).KLresult()
  }
}

case class Greedy(pointOfCube:Set[(Int, DBScanCube, Int)],cellGraph: Graph, PointsPerPartition:Int,kk:Int) {

  def getWeight(node1: Int, node2: Int): Double = {
    cellGraph.edges.getOrElse((node1, node2), 0.0)
  }

  def getSize(): Int = cellGraph.vertices.size

  def weightSum(): Double = {
    cellGraph.edges.values.sum
  }

  def sumWeights(internalSet: Set[Int], node: Int): Double = {
    var weights = 0.0
    for (i <- internalSet) {
      weights += getWeight(node, i)
    }
    weights
  }

  def idCount(cubeId: Int): Int = {
    pointOfCube.find { case (idx, _, _) => idx == cubeId } match {
      case Some((_, _, count)) => count
    }
  }

  def partitionCount(partition:mutable.Set[Int]): Int ={
    var sum = 0
    for (node <- partition) {
      pointOfCube.find { case (idx, cube, count) => idx == node } match {
        case Some((_, _, count)) =>
          sum += count
      }
    }
    sum
  }

  def getCost(partitions:mutable.Map[Int, mutable.Set[Int]]): Double ={
    var sumAll = 0.0
    for (i <- 0 until partitions.size) {
      var sum = 0.0
      for (node <- partitions(i)) {
        sum += sumWeights(partitions(i).toSet, node)
      }
      sumAll += sum
    }
    2*sumAll - weightSum
  }

  def deepCloneMap(map: mutable.Map[Int, mutable.Set[Int]]): mutable.Map[Int, mutable.Set[Int]] = {
    val newMap = mutable.Map[Int, mutable.Set[Int]]()
    for ((k, v) <- map) {
      newMap.put(k, v.clone())
    }
    newMap
  }

  def move(i:Int,j:Int,maxcost:Double,cube_par:mutable.Map[Int, Int],partitions:mutable.Map[Int, mutable.Set[Int]]):Boolean ={
    val partitionstemp = deepCloneMap(partitions)
    partitionstemp(cube_par(i)).remove(i)
    partitionstemp(j).add(i)
    val cost =  getCost(partitionstemp)    
    if(cost>maxcost) true
    else false
  }

  def KLresult(): List[Set[DBScanCube]] = {
    val partitions = mutable.Map[Int, mutable.Set[Int]]()
    val cube_par = mutable.Map[Int, Int]()  
    val k = kk
    for (i <- 0 until k) {
      partitions(i) = mutable.Set[Int]()
    }

    var partitionIndex = 0
    for (i <- 1 to getSize) {
      partitions(partitionIndex % k) += i
      cube_par(i) = partitionIndex % k
      partitionIndex += 1
    }

    var maxcost = getCost(partitions)
    for (i <- 1 to getSize) {
      for (j <- 0 until partitions.size) {
        if(idCount(i)+partitionCount(partitions(j))<PointsPerPartition){
          if(move(i,j, maxcost, cube_par, partitions)){
            partitions(cube_par(i)).remove(i)
            partitions(j).add(i)
            maxcost = getCost(partitions)
            cube_par(i) = j
          }
        }
      }
    }
    var cubepartition: List[Set[DBScanCube]] = List()
    var cubelist:Set[DBScanCube]=Set()
    var sum = 0
    var summax:Int = 0
    var summin:Int = Int.MaxValue
    for (i <- 0 until partitions.size) {
      for (node <- partitions(i)) {
        pointOfCube.find { case (idx, cube, count) => idx == node } match {
          case Some((_, cube, count)) =>
            sum += count
            cubelist += cube
        }
      }
      if(sum!=0) cubepartition = cubelist :: cubepartition
      if(sum>summax) summax = sum
      if(sum<summin) summin = sum
      cubelist = Set()
      sum = 0
    }
    cubepartition
  }
}
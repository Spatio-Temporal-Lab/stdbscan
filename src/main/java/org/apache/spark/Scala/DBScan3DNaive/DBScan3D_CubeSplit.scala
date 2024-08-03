package org.apache.spark.Scala.DBScan3DNaive

import org.apache.spark.Scala.DBScan3DNaive.DBScanLabeledPoint_3D.Flag
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.Scala.utils.partition.{CubeSplitPartition_3D, EvenSplitPartition_3D}
import org.apache.spark.Scala.utils.sample.Sample

import scala.util.control.Breaks.break

object DBScan3D_CubeSplit{
  def train(data: RDD[Vector],
            distanceEps: Double,
            timeEps: Double,
            minPoints: Int,
            numberOfPartitions: Int,
            load_balance_alpha: Double,
            x_bounding: Double,
            y_bounding: Double,
            t_bounding: Double
           ): DBScan3D_CubeSplit = {
    new DBScan3D_CubeSplit(distanceEps, timeEps, minPoints, numberOfPartitions, load_balance_alpha, x_bounding,y_bounding,t_bounding,null, null).train(data)
  }
}

class DBScan3D_CubeSplit private(val distanceEps: Double,
                                 val timeEps: Double,
                                 val minPoints: Int,
                                 val numberOfPartitions: Int,
                                 val load_balance_alpha: Double,
                                 val x_bounding: Double,
                                 val y_bounding: Double,
                                 val t_bounding: Double,
                                 @transient val partitions: List[(Int, DBScanCube)],
                                 @transient private val labeledPartitionedPoints: RDD[(Int, DBScanLabeledPoint_3D)])
  extends Serializable with Logging {
  type Margin = Set[(DBScanCube, DBScanCube, DBScanCube)]
  type ClusterID = (Int, Int)
  def minimumRectangleSize: Double = 2 * distanceEps
  def minimumHigh: Double = 2 * timeEps
  def labeledPoints: (RDD[DBScanLabeledPoint_3D], Long) = {
    (labeledPartitionedPoints.values, labeledPartitionedPoints.count()) 
  }
  def findAdjacencies(partitions: Iterable[(Int, DBScanLabeledPoint_3D)]): Set[((Int, Int), (Int, Int))] = {
    val funtionTimeBegin = System.currentTimeMillis()
    val zero = (Map[DBScanPoint_3D, ClusterID](), Set[(ClusterID, ClusterID)]())
    val partitionsMap: Map[Int, DBScanLabeledPoint_3D] = partitions.toMap
    val (_, adjacencies) = partitions.foldLeft(zero)({
      case ((seen, adajacencies), (partition, point)) => {

        if(point != null){
          if (point.flag == Flag.Noise) {
            (seen, adajacencies)
          } else if (point.flag == Flag.Core){
            val clusterId = (partition, point.cluster)
            seen.get(point) match {
              case None => (seen + (point -> clusterId), adajacencies)
              case Some(preClusterId) => {
                (seen, adajacencies + ((preClusterId, clusterId)))
              }
            }
          }else{
            val clusterId = (partition, point.cluster)
            seen.get(point) match {
              case Some(preClusterId) =>{
                if(partitionsMap(preClusterId._1).flag == Flag.Core){
                  (seen, adajacencies + ((preClusterId, clusterId)))
                }else{
                  (seen, adajacencies)
                }
              }
              case None => (seen, adajacencies)
            }
          }
        }else{
          (seen, adajacencies)
        }

      }
    })
    val functionTimeEnd = System.currentTimeMillis()
    val cost = funtionTimeBegin - functionTimeEnd
    adjacencies
  }

  def isInnerPoint(entry: (Int, DBScanLabeledPoint_3D), margins: List[(Margin, Int)]): Boolean = {
    entry match {
      case (partition, point) =>
        margins.exists {
          case (cubeSet, id) => id == partition && cubeSet.exists {
            case (inner, _, _) => inner.almostContains(point)
          }
        }
    }
  }

  val k = numberOfPartitions
  private def train(data: RDD[Vector]): DBScan3D_CubeSplit = {
    val samplePoints: Array[DBScanPoint_3D] = Sample.strict_sample(data, count = 20000)
    val localPartitions: List[Set[DBScanCube]]
    = CubeSplitPartition_3D.getPartition(samplePoints,
      x_bounding,
      y_bounding,
      t_bounding,
      k,
      load_balance_alpha
    )

    var localCubeTemp: List[Set[(DBScanCube, DBScanCube, DBScanCube)]] = List()
    for(cubeSet <- localPartitions){
      if(cubeSet != null){
        var cubeShrink : Set[(DBScanCube, DBScanCube, DBScanCube)]= Set()
        for(p <- cubeSet){
          if(p != null){
            cubeShrink += ((p.shrink(distanceEps,timeEps), p, p.shrink(-distanceEps,-timeEps)))
          }
        }
        localCubeTemp = cubeShrink :: localCubeTemp
      }

    }

    val localCube: List[(Set[(DBScanCube, DBScanCube, DBScanCube)], Int)] = localCubeTemp.zipWithIndex

    val margins: Broadcast[List[(Set[(DBScanCube, DBScanCube, DBScanCube)], Int)]] = data.context.broadcast(localCube)

    val duplicated: RDD[(Int, DBScanPoint_3D)] = data.flatMap { point =>
      val foundPoints = margins.value.flatMap { case (cubeset, id) =>
        cubeset.flatMap { case (_, _, outer) =>
          if (outer.contains(DBScanPoint_3D(point))) Some((id, DBScanPoint_3D(point)))
          else None
        }
      }
      if (foundPoints.isEmpty) {
        margins.value.map { case (_, id) => (id, DBScanPoint_3D(point)) }
      } else {
        foundPoints
      }
    }

    val duplicatedCount: Long = duplicated.count()



    val localDBScanTimeBegin = System.currentTimeMillis()
    val clustered: RDD[(Int, DBScanLabeledPoint_3D)] = duplicated
      .groupByKey()
      .flatMapValues((points: Iterable[DBScanPoint_3D]) => {
        new LocalDBScan_3D(distanceEps, timeEps, minPoints).fit(points)
      }) 
    val localDBScanTimeEnd = System.currentTimeMillis()
    val localDBScanTimeCost = localDBScanTimeBegin - localDBScanTimeEnd
   


    val marginPoints: RDD[(Int, Iterable[(Int, DBScanLabeledPoint_3D)])] = clustered.flatMap({
      case (partition, point) => {
        margins.value
          .filter({
            case (cubeSet, _) => {
              cubeSet.exists({
                case (inner, main, _) => main.contains(point) && !inner.almostContains(point)
              })
            }
          }).map({
          case (_, newPartition) => (newPartition, (partition, point))
        })
      }
    }).groupByKey()

    val adjacencies: Array[((Int, Int), (Int, Int))] = marginPoints.flatMapValues(x => findAdjacencies(x)).values.collect()
    val adjacenciesGraph = adjacencies.foldLeft(DBScanGraph_3D[ClusterID]())({
      case (graph, (from, to)) => graph.connect(from, to)
    })


    val localClusterIds = clustered.filter({
      case (_, points) => points.flag != Flag.Noise
    }).mapValues((x: DBScanLabeledPoint_3D) => x.cluster)
      .distinct()
      .collect()
      .toList


    val (total, clusterIdToGlobalId) = localClusterIds.foldLeft((0, Map[ClusterID, Int]()))({
      case ((id, map), clusterId) => {
        map.get(clusterId) match {
          case None => {
            val nextId = id + 1
            val connectedClusters: Set[(Int, Int)] = adjacenciesGraph.getConnected(clusterId) + clusterId

            val toAdd = connectedClusters.map((_, nextId)).toMap
            (nextId, map ++ toAdd)
          }
          case Some(_) => (id, map)
        }
      }
    })

    val clusterIds = data.context.broadcast(clusterIdToGlobalId)
    val labeledInner: RDD[(Int, DBScanLabeledPoint_3D)] = clustered.filter(isInnerPoint(_, margins.value))
      .map({
        case (partition, point) => {
          if (point.flag != Flag.Noise) {
            point.cluster = clusterIds.value((partition, point.cluster))
          }
          (partition, point)
        }
      })
    val labeledOuter = {
      marginPoints.flatMapValues(partition => {
        partition.foldLeft(Map[DBScanPoint_3D, DBScanLabeledPoint_3D]())({
          case (all, (partition, point)) =>

            if (point.flag != Flag.Noise) {
              point.cluster = clusterIds.value((partition, point.cluster))
            }

            all.get(point) match {
              case None => all + (point -> point)
              case Some(prev) => {

                if ((point.flag==Flag.Core&&prev.flag==Flag.Border)||(point.flag==Flag.Border&&prev.flag==Flag.Noise)) {
                  prev.flag = point.flag
                  prev.cluster = point.cluster
                }
                else if(point.flag==prev.flag){}
                else{
                  point.flag = prev.flag
                  point.cluster = prev.cluster
                }
                all+ (point -> point)
              }
            }
        }).values
      })
    }

    val finalPartition: List[(Int, DBScanCube)] = localCube.flatMap({
      case (set, index) => set.map{
        case (_, c, _) => (index, c)
      }
    })
    new DBScan3D_CubeSplit(
      distanceEps,
      timeEps,
      minPoints,
      maxPointsPerPartition,
      load_balance_alpha,
      x_bounding,
      y_bounding,
      t_bounding,
      finalPartition,
      labeledInner.union(labeledOuter))
  }
}
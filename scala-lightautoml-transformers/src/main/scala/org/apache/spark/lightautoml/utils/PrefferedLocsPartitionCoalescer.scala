package org.apache.spark.lightautoml.utils

import org.apache.spark.rdd.{PartitionCoalescer, PartitionGroup, RDD}

class PrefferedLocsPartitionCoalescer(val prefLocs: List[String]) extends PartitionCoalescer with Serializable{
  override def coalesce(maxPartitions: Int, parent: RDD[_]): Array[PartitionGroup] = {
    val parts_per_exec = (parent.partitions.length / prefLocs.size).toInt

    parent.partitions.grouped(parts_per_exec).zip(prefLocs.toIterator).flatMap {
      case (ps, prefLoc) =>
        ps.map{ p =>
          val pg = new PartitionGroup(Some(prefLoc))
          pg.partitions += p
          pg
        }
    }.toArray
  }
}

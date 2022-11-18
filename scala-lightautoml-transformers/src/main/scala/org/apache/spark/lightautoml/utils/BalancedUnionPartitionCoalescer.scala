package org.apache.spark.lightautoml.utils

import org.apache.spark.rdd.{PartitionCoalescer, PartitionGroup, RDD, UnionPartition}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col

import scala.util.Random


class BalancedUnionPartitionCoalescer extends PartitionCoalescer with Serializable {
  /**
   * This class implements a simple logic of coalescing without shuffle:
   *
   * Prerequisites:
   * 1. Takes two or more dataframes with STRICTLY the same number of partitions.
   * 2. Apply union or unionByName for all this dataframes that results to a new dataframe based on UnionRDD.
   * 3. Apply this coalescer to the resulting from the previous operation dataframe.
   *
   * How this coalescer works:
   * 1. Accepts a UnionRDD that contains of number partitions dividable by number of parent dataframes.
   * 2. Groups all partions by their respective parent Ids.
   * 3. Sorts each group by partition id in its parent RDD.
   *    (It is assumed that partitions having the same partition id and belonging
   *    to different parents are located on the same executors)
   * 4. Creates child partitions for a new RDD by regrouping parent partitions with the same Ids.
   *    (Thus each child partition depends on exactly one partition per parent)
   * */
  override def coalesce(maxPartitions: Int, parent: RDD[_]): Array[PartitionGroup] = {
    val up_arr = parent.partitions.map(_.asInstanceOf[UnionPartition[_]])
    val parent2parts = up_arr
            .groupBy(_.parentRddIndex)
            .map{case(parentRddIndex, ups) => (parentRddIndex, ups.sortBy(_.parentPartition.index))}

    val parent2size = parent2parts.map{case(parentRddIndex, ups) => (parentRddIndex, ups.length)}
    val unique_sizes = parent2size.values.toSet

    assert(
      unique_sizes.size == 1,
      s"Found differences in num of parts: $unique_sizes. Parent to parts num mapping: $parent2size"
    )

    val partsNum = unique_sizes.head

//    assert(maxPartitions <= partsNum)

    val pgs = (0 until partsNum).map(i => {
      val pg = new PartitionGroup()
      parent2parts.values.foreach(x => pg.partitions += x(i))
      pg
    })

    pgs.toArray
  }
}


#ifndef __ASTARTE_MAPPER_H__
#define __ASTARTE_MAPPER_H__

#include "default_mapper.h"
#include "legion.h"
#include "model.h"
#include "null_mapper.h"

namespace astarte {

using namespace Legion;
using namespace Mapping;

class CAShardingFunctor : public ShardingFunctor {
public:
  CAShardingFunctor(int gpus_per_node,
                    int cpus_per_node,
                    int num_nodes,
                    MachineView const &_mv);
  ~CAShardingFunctor(void);

public:
  ShardID shard(DomainPoint const &point,
                Domain const &full_space,
                const size_t total_shards);

private:
  int gpus_per_node, cpus_per_node, num_nodes;
  MachineView machine_view;
};

struct InstanceCreationLog {
  std::string task_name;
  size_t size;
  Memory memory;
  Processor processor;
};

class CAMapper : public NullMapper {
public:
  CAMapper(MapperRuntime *rt,
           Machine machine,
           Processor local,
           const char *mapper_name,
           bool _enable_control_replication,
           bool _log_instance_creation);
  ~CAMapper();
  virtual char const *get_mapper_name(void) const;
  virtual MapperSyncModel get_mapper_sync_model(void) const;

public:
  static void update_mappers(Machine machine,
                             Runtime *rt,
                             std::set<Processor> const &local_procs);
  static void register_sharding_functors(Runtime *runtime, Machine machine);
  virtual void select_task_options(const MapperContext ctx,
                                   Task const &task,
                                   TaskOptions &output);
  virtual void premap_task(const MapperContext ctx,
                           Task const &task,
                           PremapTaskInput const &input,
                           PremapTaskOutput &output);
  virtual void slice_task(const MapperContext ctx,
                          Task const &task,
                          SliceTaskInput const &input,
                          SliceTaskOutput &output);
  virtual void map_task(const MapperContext ctx,
                        Task const &task,
                        MapTaskInput const &input,
                        MapTaskOutput &output);
  virtual void map_replicate_task(const MapperContext ctx,
                                  Task const &task,
                                  MapTaskInput const &input,
                                  MapTaskOutput const &default_output,
                                  MapReplicateTaskOutput &output);
  virtual void select_task_variant(const MapperContext ctx,
                                   Task const &task,
                                   SelectVariantInput const &input,
                                   SelectVariantOutput &output);
  virtual void postmap_task(const MapperContext ctx,
                            Task const &task,
                            PostMapInput const &input,
                            PostMapOutput &output);
  virtual void select_task_sources(const MapperContext ctx,
                                   Task const &task,
                                   SelectTaskSrcInput const &input,
                                   SelectTaskSrcOutput &output);
  virtual void create_task_temporary_instance(
    const MapperContext ctx,
    const Task &task,
    CreateTaskTemporaryInput const &input,
    CreateTaskTemporaryOutput &output);
  virtual void speculate(const MapperContext ctx,
                         Task const &task,
                         SpeculativeOutput &output);
  virtual void report_profiling(const MapperContext ctx,
                                Task const &task,
                                TaskProfilingInfo const &input);
  virtual void select_sharding_functor(const MapperContext ctx,
                                       Task const &task,
                                       SelectShardingFunctorInput const &input,
                                       SelectShardingFunctorOutput &output);

public:
  virtual void map_inline(const MapperContext ctx,
                          InlineMapping constt &inline_op,
                          MapInlineInput const &input,
                          MapInlineOutput &output);
  virtual void select_inline_sources(const MapperContext ctx,
                                     InlineMapping const &inline_op,
                                     SelectInlineSrcInput const &input,
                                     SelectInlineSrcOutput &output);
  virtual void create_inline_temporary_instance(
    const MapperContext ctx,
    InlineMapping const &inline_op,
    CreateInlineTemporaryInput const &input,
    CreateInlineTemporaryOutput &output);
  virtual void report_profiling(const MapperContext ctx,
                                InlineMapping const &inline_op,
                                InlineProfilingInfo const &input);

public:
  virtual void map_copy(const MapperContext ctx,
                        Copy const &copy,
                        MapCopyInput const &input,
                        MapCopyOutput &output);
  virtual void select_copy_sources(const MapperContext ctx,
                                   Copy const &copy,
                                   SelectCopySrcInput const &input,
                                   SelectCopySrcOutput &output);
  virtual void create_copy_temporary_instance(const MapperContext ctx,
                                              Copy const &copy,
                                              CreateCopyTemporaryInput const &input,
                                              CreateCopyTemporaryOutput &output);
  virtual void speculate(const MapperContext ctx,
                         Copy const &copy,
                         SpeculativeOutput &output);
  virtual void report_profiling(const MapperContext ctx,
                                Copy const &copy,
                                CopyProfilingInfo const &input);
  virtual void select_sharding_functor(const MapperContext ctx,
                                       Copy const &copy,
                                       SelectShardingFunctorInput const &input,
                                       SelectTemplateFunctorOutput &output);

public:
  virtual void map_close(const MapperContext ctx,
                         Close const &close,
                         MapCloseInput const &input,
                         MapCloseOutput &output);
  virtual void select_close_sources(const MapperContext ctx,
                                    Close const &close,
                                    SelectCloseSrcInput const &input,
                                    selectCloseSrcOutput &output);
}

}


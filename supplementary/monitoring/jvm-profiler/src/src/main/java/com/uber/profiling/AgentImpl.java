/*
 * Copyright (c) 2018 Uber Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.uber.profiling;

import com.uber.profiling.profilers.CpuAndMemoryProfiler;
import com.uber.profiling.profilers.IOProfiler;
import com.uber.profiling.profilers.MethodArgumentCollector;
import com.uber.profiling.profilers.MethodArgumentProfiler;
import com.uber.profiling.profilers.MethodDurationCollector;
import com.uber.profiling.profilers.MethodDurationProfiler;
import com.uber.profiling.profilers.ProcessInfoProfiler;
import com.uber.profiling.profilers.StacktraceCollectorProfiler;
import com.uber.profiling.profilers.StacktraceReporterProfiler;
import com.uber.profiling.profilers.ThreadInfoProfiler;
import com.uber.profiling.transformers.JavaAgentFileTransformer;
import com.uber.profiling.transformers.MethodProfilerStaticProxy;
import com.uber.profiling.util.*;

import java.lang.instrument.Instrumentation;
import java.lang.Thread;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

public class AgentImpl {
    public static final String VERSION = "1.0.0";
    
    private static final AgentLogger logger = AgentLogger.getLogger(AgentImpl.class.getName());

    private static final int MAX_THREAD_POOL_SIZE = 2;

    private boolean started = false;

    public void run(Arguments arguments, Instrumentation instrumentation, Collection<AutoCloseable> objectsToCloseOnShutdown) {
        if (arguments.isNoop()) {
            logger.info("Agent noop is true, do not run anything");
            return;
        }
        
        Reporter reporter = arguments.getReporter();

        String processUuid = UUID.randomUUID().toString();

        String appId = null;
        String executorID = SparkUtils.getSparkEnvExecutorId();

        String appIdVariable = arguments.getAppIdVariable();
        if (appIdVariable != null && !appIdVariable.isEmpty()) {
            appId = System.getenv(appIdVariable);
        }
        
        if (appId == null || appId.isEmpty()) {
            appId = SparkUtils.probeAppId(arguments.getAppIdRegex());
        }


        String cmdline = ProcFileUtils.getCmdline();
        logger.info("cmdline: " + cmdline);

//        int i = 0;
//        while (executorID == null && i < 50) {
//            logger.info("executorID == null, trying to get executorID.");
//            try {
//                Thread.sleep(1000);
//            } catch (InterruptedException e) {
//                throw new RuntimeException(e);
//            }
//            executorID = SparkUtils.getSparkEnvExecutorId();
//            i++;
//        }
        if (executorID == null) {
            executorID = SparkUtils.getSparkEnvExecutorId();
        }

        if (!arguments.getDurationProfiling().isEmpty()
                || !arguments.getArgumentProfiling().isEmpty()) {
            instrumentation.addTransformer(new JavaAgentFileTransformer(arguments.getDurationProfiling(),
                    arguments.getArgumentProfiling()), true);

            Set<String> loadedClasses = Arrays.stream(instrumentation.getAllLoadedClasses())
                    .map(Class::getName).collect(Collectors.toSet());

            Set<String> tobeReloadClasses = arguments.getDurationProfiling().stream()
                    .map(ClassAndMethod::getClassName).collect(Collectors.toSet());

            tobeReloadClasses.addAll(arguments.getArgumentProfiling().stream()
                    .map(ClassMethodArgument::getClassName).collect(Collectors.toSet()));

            tobeReloadClasses.retainAll(loadedClasses);

            tobeReloadClasses.forEach(clazz -> {
                try {
                    instrumentation.retransformClasses(Class.forName(clazz));
                    logger.info("Reload class [" + clazz + "] success.");
                } catch (Exception e) {
                    logger.warn("Reload class [" + clazz + "] failed.", e);
                }
            });
        }

        List<Profiler> profilers = createProfilers(reporter, arguments, processUuid, appId, executorID);

        ProfilerGroup profilerGroup = startProfilers(profilers);

        Thread shutdownHook = new Thread(new ShutdownHookRunner(profilerGroup.getPeriodicProfilers(), Arrays.asList(reporter), objectsToCloseOnShutdown));
        Runtime.getRuntime().addShutdownHook(shutdownHook);
    }

    public ProfilerGroup startProfilers(Collection<Profiler> profilers) {
        if (started) {
            logger.warn("Profilers already started, do not start it again");
            return new ProfilerGroup(new ArrayList<>(), new ArrayList<>());
        }

        List<Profiler> oneTimeProfilers = new ArrayList<>();
        List<Profiler> periodicProfilers = new ArrayList<>();

        for (Profiler profiler : profilers) {
            if (profiler.getIntervalMillis() == 0) {
                oneTimeProfilers.add(profiler);
            } else if (profiler.getIntervalMillis() > 0) {
                periodicProfilers.add(profiler);
            } else {
                logger.log(String.format("Ignored profiler %s due to its invalid interval %s", profiler, profiler.getIntervalMillis()));
            }
        }

        for (Profiler profiler : oneTimeProfilers) {
            try {
                profiler.profile();
                logger.info("Finished one time profiler: " + profiler);
            } catch (Throwable ex) {
                logger.warn("Failed to run one time profiler: " + profiler, ex);
            }
        }

        for (Profiler profiler : periodicProfilers) {
            try {
                profiler.profile();
                logger.info("Ran periodic profiler (first run): " + profiler);
            } catch (Throwable ex) {
                logger.warn("Failed to run periodic profiler (first run): " + profiler, ex);
            }
        }
        
        scheduleProfilers(periodicProfilers);
        started = true;

        return new ProfilerGroup(oneTimeProfilers, periodicProfilers);
    }

    private List<Profiler> createProfilers(Reporter reporter, Arguments arguments, String processUuid, String appId, String executorID) {
        String tag = arguments.getTag();
        String cluster = arguments.getCluster();
        long metricInterval = arguments.getMetricInterval();

        List<Profiler> profilers = new ArrayList<>();

        CpuAndMemoryProfiler cpuAndMemoryProfiler = new CpuAndMemoryProfiler(reporter);
        cpuAndMemoryProfiler.setTag(tag);
        cpuAndMemoryProfiler.setCluster(cluster);
        cpuAndMemoryProfiler.setIntervalMillis(metricInterval);
        cpuAndMemoryProfiler.setProcessUuid(processUuid);
        cpuAndMemoryProfiler.setAppId(appId);
        cpuAndMemoryProfiler.setExecutorId(executorID);

        profilers.add(cpuAndMemoryProfiler);

        ProcessInfoProfiler processInfoProfiler = new ProcessInfoProfiler(reporter);
        processInfoProfiler.setTag(tag);
        processInfoProfiler.setCluster(cluster);
        processInfoProfiler.setProcessUuid(processUuid);
        processInfoProfiler.setAppId(appId);
        processInfoProfiler.setExecutorId(executorID);

        profilers.add(processInfoProfiler);

        if (arguments.isThreadProfiling()) {
            ThreadInfoProfiler threadInfoProfiler = new ThreadInfoProfiler(reporter);
            threadInfoProfiler.setTag(tag);
            threadInfoProfiler.setCluster(cluster);
            threadInfoProfiler.setIntervalMillis(metricInterval);
            threadInfoProfiler.setProcessUuid(processUuid);
            threadInfoProfiler.setAppId(appId);

            profilers.add(threadInfoProfiler);
        }

        if (!arguments.getDurationProfiling().isEmpty()) {
            ClassAndMethodLongMetricBuffer classAndMethodMetricBuffer = new ClassAndMethodLongMetricBuffer();

            MethodDurationProfiler methodDurationProfiler = new MethodDurationProfiler(classAndMethodMetricBuffer, reporter);
            methodDurationProfiler.setTag(tag);
            methodDurationProfiler.setCluster(cluster);
            methodDurationProfiler.setIntervalMillis(metricInterval);
            methodDurationProfiler.setProcessUuid(processUuid);
            methodDurationProfiler.setAppId(appId);
            methodDurationProfiler.setExecutorId(executorID);

            MethodDurationCollector methodDurationCollector = new MethodDurationCollector(classAndMethodMetricBuffer);
            MethodProfilerStaticProxy.setCollector(methodDurationCollector);

            profilers.add(methodDurationProfiler);
        }

        if (!arguments.getArgumentProfiling().isEmpty()) {
            ClassMethodArgumentMetricBuffer classAndMethodArgumentBuffer = new ClassMethodArgumentMetricBuffer();

            MethodArgumentProfiler methodArgumentProfiler = new MethodArgumentProfiler(classAndMethodArgumentBuffer, reporter);
            methodArgumentProfiler.setTag(tag);
            methodArgumentProfiler.setCluster(cluster);
            methodArgumentProfiler.setIntervalMillis(metricInterval);
            methodArgumentProfiler.setProcessUuid(processUuid);
            methodArgumentProfiler.setAppId(appId);
            methodArgumentProfiler.setExecutorId(executorID);

            MethodArgumentCollector methodArgumentCollector = new MethodArgumentCollector(classAndMethodArgumentBuffer);
            MethodProfilerStaticProxy.setArgumentCollector(methodArgumentCollector);

            profilers.add(methodArgumentProfiler);
        }
        
        if (arguments.getSampleInterval() > 0) {
            StacktraceMetricBuffer stacktraceMetricBuffer = new StacktraceMetricBuffer();

            StacktraceCollectorProfiler stacktraceCollectorProfiler = new StacktraceCollectorProfiler(stacktraceMetricBuffer, AgentThreadFactory.NAME_PREFIX);
            stacktraceCollectorProfiler.setIntervalMillis(arguments.getSampleInterval());
                    
            StacktraceReporterProfiler stacktraceReporterProfiler = new StacktraceReporterProfiler(stacktraceMetricBuffer, reporter);
            stacktraceReporterProfiler.setTag(tag);
            stacktraceReporterProfiler.setCluster(cluster);
            stacktraceReporterProfiler.setIntervalMillis(metricInterval);
            stacktraceReporterProfiler.setProcessUuid(processUuid);
            stacktraceReporterProfiler.setAppId(appId);
            stacktraceReporterProfiler.setExecutorId(executorID);

            profilers.add(stacktraceCollectorProfiler);
            profilers.add(stacktraceReporterProfiler);
        }

        if (arguments.isIoProfiling()) {
            IOProfiler ioProfiler = new IOProfiler(reporter);
            ioProfiler.setTag(tag);
            ioProfiler.setCluster(cluster);
            ioProfiler.setIntervalMillis(metricInterval);
            ioProfiler.setProcessUuid(processUuid);
            ioProfiler.setAppId(appId);
            ioProfiler.setExecutorId(executorID);

            profilers.add(ioProfiler);
        }
        
        return profilers;
    }

    private void scheduleProfilers(Collection<Profiler> profilers) {
        int threadPoolSize = Math.min(profilers.size(), MAX_THREAD_POOL_SIZE);
        ScheduledExecutorService scheduledExecutorService = Executors.newScheduledThreadPool(threadPoolSize, new AgentThreadFactory());

        for (Profiler profiler : profilers) {
            if (profiler.getIntervalMillis() < Arguments.MIN_INTERVAL_MILLIS) {
                throw new RuntimeException("Interval too short for profiler: " + profiler + ", must be at least " + Arguments.MIN_INTERVAL_MILLIS);
            }
            
            ProfilerRunner worker = new ProfilerRunner(profiler);
            scheduledExecutorService.scheduleAtFixedRate(worker, 0, profiler.getIntervalMillis(), TimeUnit.MILLISECONDS);
            logger.info(String.format("Scheduled profiler %s with interval %s millis", profiler, profiler.getIntervalMillis()));
        }
    }
}

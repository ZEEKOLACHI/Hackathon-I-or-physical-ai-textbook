---
id: ch-1-02
title: ROS 2 Fundamentals
sidebar_position: 2
difficulty: beginner
estimated_time: 50
prerequisites: [ch-1-01]
---

# ROS 2 Fundamentals

> "In robotics, communication isn't just about sending messages—it's about coordinating action in an uncertain, time-critical world." — Open Robotics

The Robot Operating System has become the *lingua franca* of robotics research and development. Understanding ROS 2 is not just about learning an API—it's about understanding how modern robotic systems are architected, how distributed components communicate, and how we manage complexity in real-time systems.

## The Evolution of Robot Middleware

### Why Do Robots Need Middleware?

Imagine building a humanoid robot from scratch. You need to:
- Read data from cameras, LiDAR, IMUs, and force sensors
- Process that data through perception algorithms
- Run planning algorithms to decide what to do
- Send commands to dozens of motor controllers
- Handle failures, monitor health, and log data
- All while ensuring nothing crashes and timing constraints are met

Without middleware, you would write custom communication code for every connection. A robot with 20 components would need potentially 190 direct connections. This is unmaintainable chaos.

**Middleware provides:**
- Standardized communication patterns
- Discovery mechanisms so components find each other
- Hardware abstraction so algorithms don't depend on specific sensors
- Debugging and visualization tools
- A package ecosystem to avoid reinventing wheels

### The ROS Legacy

The original Robot Operating System (ROS 1), created at Willow Garage in 2007, revolutionized robotics research. It solved the fragmentation problem—suddenly, researchers worldwide could share code, collaborate on packages, and build on each other's work.

But ROS 1 had limitations:

| ROS 1 Limitation | Real-World Impact |
|------------------|-------------------|
| **Single point of failure** | If the master crashes, the entire system stops |
| **No real-time support** | Can't guarantee timing for safety-critical control |
| **No security** | Anyone on the network can send any message |
| **Linux-only** | Limits commercial deployment options |
| **Global namespace** | Multi-robot coordination is awkward |

By 2015, these limitations became critical as robots moved from research labs to factories, hospitals, and homes.

### ROS 2: A Modern Foundation

ROS 2 represents a ground-up redesign built on decades of lessons from ROS 1. The key insight: rather than building custom communication infrastructure, ROS 2 adopts **DDS (Data Distribution Service)**—an industry-standard middleware used in military systems, financial trading, and aerospace.

```
ROS 1 Architecture              ROS 2 Architecture
─────────────────               ──────────────────

┌─────────────┐                 ┌─────────────┐
│   Node A    │                 │   Node A    │
└──────┬──────┘                 └──────┬──────┘
       │ Custom Protocol                │ DDS
       ▼                                ▼
┌─────────────┐                 ┌─────────────────────┐
│   Master    │                 │  DDS Discovery      │
│  (SPOF!)    │                 │  (Distributed)      │
└──────┬──────┘                 └──────────┬──────────┘
       │                                   │
       ▼                                   ▼
┌─────────────┐                 ┌─────────────┐
│   Node B    │                 │   Node B    │
└─────────────┘                 └─────────────┘
```

## Core Communication Paradigms

ROS 2 provides three fundamental communication patterns, each designed for different use cases.

### Topics: The Publish-Subscribe Pattern

Topics are the workhorse of ROS 2 communication. They implement a **publish-subscribe pattern** where:
- Publishers send messages without knowing who receives them
- Subscribers receive messages without knowing who sends them
- The middleware handles discovery and routing

**Why Publish-Subscribe?**

This decoupling is powerful:

| Benefit | Description |
|---------|-------------|
| **Loose coupling** | Publishers and subscribers don't depend on each other |
| **Dynamic discovery** | Components can join and leave at runtime |
| **Multiple subscribers** | One sensor can feed many algorithms |
| **Multiple publishers** | Data can be aggregated from multiple sources |
| **Scalability** | Adding components doesn't require code changes |

**When to Use Topics:**
- Continuous sensor data streams (cameras, LiDAR, IMU)
- State information that updates regularly (robot pose, joint states)
- Commands that are "fire and forget" (velocity commands)
- Any data where you want many-to-many communication

**Topic Semantics:**

Topics have a **name** (like `/camera/image_raw`) and a **message type** (like `sensor_msgs/msg/Image`). The type defines the data structure and ensures type safety—you can't accidentally send an image on a topic expecting a point cloud.

### Services: Synchronous Request-Response

Not all communication fits the pub-sub model. Sometimes you need to:
- Ask a question and wait for an answer
- Trigger an action and confirm it completed
- Query the current state of a component

Services provide **synchronous request-response** communication:

```
Client                          Server
──────                          ──────
   │                              │
   │──── Request ────────────────►│
   │                              │ (Processing)
   │◄─── Response ───────────────│
   │                              │
```

**Service Characteristics:**
- One-to-one communication (one client, one server per call)
- Blocking by default (client waits for response)
- Guaranteed delivery (or timeout/failure)
- Stateless—each call is independent

**When to Use Services:**
- Configuration queries ("What is your current parameter?")
- One-shot commands ("Save the map now")
- State transitions ("Switch to manual mode")
- Computations ("Plan a path from A to B")

**The Service Anti-Pattern:**

A common mistake is using services for things that should be topics. If you find yourself:
- Calling a service repeatedly at a fixed rate
- Ignoring the response
- Having multiple clients that all need the same data

...you probably want a topic instead.

### Actions: Long-Running Tasks with Feedback

Some robot behaviors take time—navigating to a goal, performing a manipulation task, or executing a complex motion. For these, neither topics nor services are ideal:

- Topics can't represent "start a task, monitor progress, get final result"
- Services block, which is problematic for long operations

**Actions** solve this with a richer protocol:

```
Action Client                    Action Server
─────────────                    ─────────────
      │                                │
      │──── Goal ─────────────────────►│
      │◄─── Goal Accepted ────────────│
      │                                │ (Executing)
      │◄─── Feedback ─────────────────│
      │◄─── Feedback ─────────────────│
      │◄─── Feedback ─────────────────│
      │                                │
      │◄─── Result ───────────────────│
      │                                │
```

**Action Features:**
- **Goal submission**: Client sends desired outcome
- **Goal acceptance**: Server can accept or reject
- **Feedback**: Periodic progress updates during execution
- **Result**: Final outcome when complete
- **Cancellation**: Client can request stopping

**When to Use Actions:**
- Navigation goals ("Go to the kitchen")
- Manipulation tasks ("Pick up the cup")
- Complex motions ("Perform a walking gait cycle")
- Anything that takes seconds to minutes and benefits from progress updates

## Understanding DDS and QoS

### What is DDS?

The **Data Distribution Service** is an OMG (Object Management Group) standard for real-time publish-subscribe communication. It's used in:
- Military systems (ships, aircraft, battlefield networks)
- Financial trading (low-latency market data distribution)
- Aerospace (air traffic control, satellite systems)
- Healthcare (hospital monitoring systems)

ROS 2 doesn't implement DDS—it uses existing DDS implementations as a foundation. Common choices:

| Implementation | Characteristics |
|----------------|-----------------|
| **Fast DDS** (eProsima) | Default in ROS 2, open-source, feature-rich |
| **Cyclone DDS** (Eclipse) | High performance, lightweight, Eclipse Foundation backed |
| **Connext DDS** (RTI) | Commercial, extremely performant, used in critical systems |

### Quality of Service (QoS)

DDS introduces a powerful concept: **Quality of Service policies** that configure how messages are handled.

**Key QoS Policies:**

| Policy | Options | Meaning |
|--------|---------|---------|
| **Reliability** | RELIABLE, BEST_EFFORT | Guarantee delivery or accept drops? |
| **Durability** | VOLATILE, TRANSIENT_LOCAL | Keep messages for late joiners? |
| **History** | KEEP_LAST(n), KEEP_ALL | How many messages to buffer? |
| **Deadline** | Duration | Expected update rate |
| **Lifespan** | Duration | How long messages are valid |

**QoS Compatibility:**

Publishers and subscribers must have compatible QoS. A common issue: a RELIABLE publisher and BEST_EFFORT subscriber won't connect. ROS 2 provides tools to diagnose these mismatches.

**Choosing QoS:**

| Use Case | Recommended QoS |
|----------|-----------------|
| **Sensor data** | BEST_EFFORT, KEEP_LAST(1) — latest data matters most |
| **Commands** | RELIABLE, KEEP_LAST(10) — don't lose commands |
| **State** | RELIABLE, TRANSIENT_LOCAL — late joiners get current state |
| **Images** | BEST_EFFORT, KEEP_LAST(1) — too big for reliable queuing |

## The Node Graph: Thinking in Systems

ROS 2 systems are **graphs** of interconnected nodes. Understanding this graph-based architecture is essential for designing maintainable systems.

### Node Design Principles

**Single Responsibility:**
Each node should do one thing well. Don't create "god nodes" that handle everything. Benefits:
- Easier to test in isolation
- Easier to replace or upgrade components
- Better fault isolation
- Clearer system understanding

**Composability:**
Design nodes to be combined in different configurations. A perception node shouldn't assume anything about the planner that will use its output.

**Interface Stability:**
Topic names and message types are your API. Changing them breaks downstream code. Design interfaces carefully and version them appropriately.

### System Composition Patterns

**Pipeline Pattern:**
Linear data flow through processing stages.
```
Sensor → Preprocessor → Detector → Tracker → Behavior
```

**Hierarchical Pattern:**
Nested controllers with increasing abstraction.
```
High-Level Planner
       │
       ▼
Motion Planner
       │
       ▼
Low-Level Controller
       │
       ▼
    Hardware
```

**Parallel Pattern:**
Multiple algorithms processing the same input.
```
        ┌─► Visual Odometry ─┐
Camera ─┼─► Object Detection ─┼─► Fusion
        └─► Semantic Segmentation ─┘
```

## Real-Time Considerations

Physical AI systems have hard timing requirements. A control loop that usually runs at 1kHz but occasionally takes 50ms will cause jerky, potentially dangerous motion.

### The Real-Time Challenge

"Real-time" doesn't mean "fast"—it means **predictable**. A real-time system:
- Guarantees worst-case execution time
- Avoids unbounded operations (dynamic memory allocation, blocking I/O)
- Uses priority-based scheduling
- Minimizes jitter (variation in timing)

### ROS 2 Real-Time Support

ROS 2 was designed with real-time in mind:

| Feature | Purpose |
|---------|---------|
| **DDS** | Many DDS implementations are real-time capable |
| **rcl (C library)** | Core library avoids dynamic allocation |
| **Executors** | Control over callback scheduling |
| **Lifecycle nodes** | Predictable startup and shutdown |
| **Real-time-safe allocators** | Pre-allocated memory pools |

### Practical Real-Time Tips

1. **Use BEST_EFFORT for sensor data**: Reliable delivery introduces latency
2. **Pre-allocate messages**: Avoid dynamic allocation in hot paths
3. **Use intra-process communication**: Zero-copy when nodes share a process
4. **Profile your callbacks**: Identify and fix timing outliers
5. **Separate real-time from non-real-time**: Keep logging, visualization in separate processes

## Lifecycle Management

Production robots need controlled startup and shutdown. You can't have a planning node crash because the sensor node wasn't ready yet.

### Managed Nodes

ROS 2 introduces **lifecycle nodes** with well-defined states:

```
         ┌─────────────────────────────────────────┐
         │                                         │
         ▼                                         │
    ┌─────────┐    configure    ┌──────────┐      │
────► Unconfigured ─────────────► Inactive │      │
    └─────────┘                 └────┬─────┘      │
         ▲                           │            │
         │ cleanup              activate         │
         │                           │            │
    ┌────┴─────┐                    ▼            │
    │ Finalized │◄─ shutdown ─ ┌──────────┐      │
    └──────────┘               │  Active  │──────┘
                               └──────────┘   deactivate
```

**State Descriptions:**

| State | Description |
|-------|-------------|
| **Unconfigured** | Node exists but has no resources |
| **Inactive** | Resources allocated, ready to activate |
| **Active** | Fully operational, processing data |
| **Finalized** | Shutting down, releasing resources |

### Benefits of Lifecycle Management

- **Deterministic startup**: Configure all nodes, then activate in order
- **Graceful degradation**: Deactivate failing components without full restart
- **Testing**: Validate configuration before activation
- **Hot-swapping**: Replace components at runtime

## The ROS 2 Ecosystem

### Package Philosophy

ROS 2's power comes from its ecosystem—thousands of packages solving common problems:

**Standard Packages:**
- `geometry_msgs`: Points, vectors, poses, transforms
- `sensor_msgs`: Camera images, laser scans, point clouds, IMU data
- `nav_msgs`: Odometry, occupancy grids, paths
- `visualization_msgs`: Markers for RViz display

**Major Frameworks:**
- `Nav2`: Complete navigation stack
- `MoveIt 2`: Motion planning for manipulation
- `ros2_control`: Hardware interface and controller framework
- `image_pipeline`: Camera calibration and processing

### Development Tools

| Tool | Purpose |
|------|---------|
| **RViz 2** | 3D visualization of robot state and sensor data |
| **rqt** | Plugin-based GUI tools (plots, topic monitor, service caller) |
| **ros2 bag** | Record and playback message streams |
| **launch** | Orchestrate starting multiple nodes with configuration |
| **colcon** | Build system for ROS 2 workspaces |

## Building Mental Models

### The Message Flow Perspective

Think of your robot as a message processing pipeline:

1. **Sensors produce messages** at their native rates
2. **Processors consume and transform** messages
3. **Controllers produce commands** based on processed state
4. **Actuators consume commands** and affect the world
5. **The world changes** and sensors observe new state

Every design decision affects this flow. Bottlenecks appear where processing can't keep up. Latency accumulates through the pipeline. Understanding the flow helps diagnose problems.

### The Graph Perspective

Your system is a graph where:
- Nodes are **vertices** with computation
- Topics are **edges** carrying data
- The graph topology determines information flow

Tools like `rqt_graph` visualize this structure. A healthy graph is understandable at a glance. A tangled graph suggests architectural problems.

### The Time Perspective

Messages carry timestamps. Coordinating time-stamped data is crucial:
- **Clock synchronization**: All nodes agree on current time
- **Time travel**: Historical data in bags, simulation time
- **Transform trees**: Pose relationships at specific times

The `tf2` library manages time-stamped transformations between coordinate frames—essential for any perception or control task.

## Summary

ROS 2 provides the communication infrastructure for modern robotics:

**Core Concepts:**
- **Nodes** are modular processing units
- **Topics** enable pub-sub communication for sensor streams and continuous data
- **Services** provide request-response for queries and one-shot commands
- **Actions** handle long-running tasks with feedback and cancellation

**Key Design Principles:**
- **DDS foundation** provides reliability, security, and real-time capabilities
- **QoS policies** configure communication characteristics
- **Lifecycle nodes** enable deterministic system management
- **Graph-based architecture** promotes modularity and understanding

**Ecosystem:**
- Standardized message types for interoperability
- Extensive package library to avoid reinventing wheels
- Powerful development and debugging tools

In the next chapter, we'll put ROS 2 to work in simulation environments, where we can develop and test robot software safely before deploying to hardware.

## Further Reading

- **Open Robotics** — [ROS 2 Documentation](https://docs.ros.org/en/humble/)
- **OMG** — [DDS Specification](https://www.omg.org/spec/DDS/)
- **Maruyama et al.** — "Exploring the Performance of ROS 2" (EMSOFT 2016)
- **Thomas et al.** — "ROS 2: The Future of Robotics Middleware" (IEEE RA-M 2022)

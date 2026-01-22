---
id: ch-4-11
title: Force Control
sidebar_position: 2
difficulty: advanced
estimated_time: 35
prerequisites: [ch-4-10]
---

# Force Control

Force control enables robots to interact safely and precisely with their environment.

## Impedance Control

```python
class ImpedanceController:
    """
    Impedance control: regulate dynamic relationship
    between force and motion.
    """
    def __init__(self, mass, damping, stiffness):
        self.M = mass      # Virtual inertia
        self.B = damping   # Virtual damping
        self.K = stiffness # Virtual stiffness

    def compute(self, x_des, x, x_dot, f_ext):
        """
        Compute desired acceleration based on
        impedance model.
        """
        # Impedance equation: M*x_ddot + B*x_dot + K*x = f_ext
        x_ddot = (f_ext - self.B * x_dot - self.K * (x - x_des)) / self.M
        return x_ddot
```

## Admittance Control

```python
class AdmittanceController:
    """
    Admittance control: force input, motion output.
    """
    def __init__(self, mass, damping, stiffness, dt=0.001):
        self.M = mass
        self.B = damping
        self.K = stiffness
        self.dt = dt

        self.x = 0
        self.x_dot = 0

    def compute(self, f_ext, x_des):
        """
        Compute position adjustment based on external force.
        """
        # Admittance model: f = M*x_ddot + B*x_dot + K*x
        x_ddot = (f_ext - self.B * self.x_dot - self.K * (self.x - x_des)) / self.M

        # Integrate
        self.x_dot += x_ddot * self.dt
        self.x += self.x_dot * self.dt

        return self.x
```

## Hybrid Position/Force Control

```python
class HybridController:
    """
    Control position in some directions,
    force in others.
    """
    def __init__(self):
        self.position_controller = PIDController(1.0, 0.1, 0.05)
        self.force_controller = PIDController(0.001, 0.0001, 0)

        # Selection matrix: 1 = position, 0 = force
        self.S = np.diag([1, 1, 0, 1, 1, 1])  # Force control in Z

    def compute(self, x_des, f_des, x, f):
        pos_control = self.position_controller.compute(x_des, x)
        force_control = self.force_controller.compute(f_des, f)

        return self.S @ pos_control + (np.eye(6) - self.S) @ force_control
```

## Summary

- Impedance control regulates the force-motion relationship
- Admittance control converts force to motion
- Hybrid control combines position and force control
- Essential for safe human-robot interaction

## Further Reading

- Siciliano, B. "Robotics: Modelling, Planning and Control"

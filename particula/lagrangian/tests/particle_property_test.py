"""test module for particle_property.py"""
import torch
from particula.lagrangian import particle_property


def test_radius_vector():
    """Test calculating the radius vector."""
    mass = torch.tensor([10.0, 20.0, 30.0])
    density = torch.tensor([2.0, 3.0, 4.0])
    expected_radius = torch.tensor(
        [1.0608, 1.1675, 1.2143])
    result = particle_property.radius_calculation(mass, density)
    assert torch.allclose(result, expected_radius, atol=1e-4)


def test_friction_factor_runs():
    """testing if the friction factor function runs
    correct values are checked by untls tests"""
    radius = torch.tensor([1.0, 2.0, 3.0])
    temperature = 300.0
    pressure = 1e5
    result = particle_property.friction_factor_wrapper(
        radius_meter=radius,
        temperature_kelvin=temperature,
        pressure_pascal=pressure,
    )
    assert result.shape == radius.shape


def test_generate_mass():
    """test generating masses"""
    mean_radius = 100  # example value
    std_dev_radius = 15  # example value
    density = torch.tensor(1e3)  # example density
    num_particles = 1000

    masses = particle_property.generate_particle_masses(
        mean_radius=mean_radius,
        std_dev_radius=std_dev_radius,
        density=density,
        num_particles=num_particles
    )
    assert masses.shape[0] == num_particles
    assert torch.all(masses >= 0)  # Masses should be non-negative


def test_thermal_speed_calculation():
    """test calculating thermal velocity
    should check the values"""
    temperature = 273.0  # example value
    mass = particle_property.mass_calculation(radius=torch.tensor([1e-6]),
                                  density=torch.tensor([1e3]))
    expected_velocity = torch.tensor([0.0015])
    result = particle_property.thermal_speed(
        temperature_kelvin=temperature,
        mass_kg=mass,
    )
    assert torch.allclose(result, expected_velocity, atol=1e-3)


def test_speed_calculation():
    """test calculating speed"""
    velocity = torch.tensor([1.0, 2.0, 3.0])
    expected_speed = torch.tensor(3.7417)
    result = particle_property.speed(velocity)
    assert torch.allclose(result, expected_speed, atol=1e-4)


def test_random_thermal_velocity():
    """test generating thermal velocity"""
    temperature = 300  # example value
    mass = torch.tensor(1e-20)  # example mass
    num_particles = 1000

    velocities = particle_property.random_thermal_velocity(
        temperature_kelvin=temperature,
        mass_kg=mass,
        number_of_particles=num_particles
    )
    assert velocities.shape[1] == num_particles

// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <chrono>
#include <thread>

#include "gtest/gtest.h"
#include "rclcpp/rclcpp.hpp"
#include "nav2_mppi_controller/motion_models.hpp"
#include "nav2_mppi_controller/models/state.hpp"
#include "nav2_mppi_controller/models/control_sequence.hpp"
#include "nav2_mppi_controller/tools/utils.hpp"

// Tests motion models

using namespace mppi;  // NOLINT

TEST(MotionModelTests, DiffDriveTest)
{
  models::ControlSequence control_sequence;
  models::State state;
  int batches = 1000;
  int timesteps = 50;
  control_sequence.reset(timesteps);  // populates with zeros
  state.reset(batches, timesteps);  // populates with zeros
  std::unique_ptr<DiffDriveMotionModel> model =
    std::make_unique<DiffDriveMotionModel>();

  // Check that predict properly populates the trajectory velocities with the control velocities
  state.cvx = 10 * Eigen::ArrayXXf::Ones(batches, timesteps);
  state.cvy = 5 * Eigen::ArrayXXf::Ones(batches, timesteps);
  state.cwz = 1 * Eigen::ArrayXXf::Ones(batches, timesteps);

  // Manually set state index 0 from initial conditions which would be the speed of the robot
  state.vx.col(0) = 10;
  state.wz.col(0) = 1;

  model->predict(state);

  EXPECT_TRUE(state.vx.isApprox(state.cvx));
  EXPECT_TRUE(state.vy.isApprox(Eigen::ArrayXXf::Zero(batches, timesteps)));  // non-holonomic
  EXPECT_TRUE(state.wz.isApprox(state.cwz));

  // Check that application of constraints are empty for Diff Drive
  for (unsigned int i = 0; i != control_sequence.vx.rows(); i++) {
    control_sequence.vx(i) = i * i * i;
    control_sequence.wz(i) = i * i * i;
  }

  models::ControlSequence initial_control_sequence = control_sequence;
  model->applyConstraints(control_sequence);
  EXPECT_TRUE(initial_control_sequence.vx.isApprox(control_sequence.vx));
  EXPECT_TRUE(initial_control_sequence.vy.isApprox(control_sequence.vy));
  EXPECT_TRUE(initial_control_sequence.wz.isApprox(control_sequence.wz));

  // Check that Diff Drive is properly non-holonomic
  EXPECT_EQ(model->isHolonomic(), false);

  // Check it cleanly destructs
  model.reset();
}

TEST(MotionModelTests, OmniTest)
{
  models::ControlSequence control_sequence;
  models::State state;
  int batches = 1000;
  int timesteps = 50;
  control_sequence.reset(timesteps);  // populates with zeros
  state.reset(batches, timesteps);  // populates with zeros

  auto node = std::make_shared<nav2::LifecycleNode>("omni_test_node");
  std::string name = "test";
  ParametersHandler param_handler(node, name);
  // Explicitly disable elliptic limits so applyConstraints is a no-op (legacy behavior)
  node->declare_parameter(name + ".omni.elliptic_limits", rclcpp::ParameterValue(false));

  std::unique_ptr<OmniMotionModel> model =
    std::make_unique<OmniMotionModel>();
  model->initialize(&param_handler, name + ".omni");

  // Check that predict properly populates the trajectory velocities with the control velocities
  state.cvx = 10 * Eigen::ArrayXXf::Ones(batches, timesteps);
  state.cvy = 5 * Eigen::ArrayXXf::Ones(batches, timesteps);
  state.cwz = 1 * Eigen::ArrayXXf::Ones(batches, timesteps);

  // Manually set state index 0 from initial conditions which would be the speed of the robot
  state.vx.col(0) = 10;
  state.vy.col(0) = 5;
  state.wz.col(0) = 1;

  model->predict(state);

  EXPECT_TRUE(state.vx.isApprox(state.cvx));
  EXPECT_TRUE(state.vy.isApprox(state.cvy));  // holonomic
  EXPECT_TRUE(state.wz.isApprox(state.cwz));

  // With elliptic_limits disabled, applyConstraints must be a no-op (legacy rectangular space)
  for (unsigned int i = 0; i != control_sequence.vx.rows(); i++) {
    control_sequence.vx(i) = i * i * i;
    control_sequence.vy(i) = i * i * i;
    control_sequence.wz(i) = i * i * i;
  }

  models::ControlConstraints c{0.5f, -0.35f, 0.5f, 1.9f, 3.0f, -3.0f, -3.0f, 3.0f, 3.5f};
  model->setConstraints(c, 0.05f, 0.0f, 0.0f, 0.0f, false);

  models::ControlSequence initial_control_sequence = control_sequence;
  model->applyConstraints(control_sequence);
  EXPECT_TRUE(initial_control_sequence.vx.isApprox(control_sequence.vx));
  EXPECT_TRUE(initial_control_sequence.vy.isApprox(control_sequence.vy));
  EXPECT_TRUE(initial_control_sequence.wz.isApprox(control_sequence.wz));

  // Check that Omni Drive is properly holonomic
  EXPECT_EQ(model->isHolonomic(), true);

  // Check it cleanly destructs
  model.reset();
}

TEST(MotionModelTests, OmniEllipticForwardClampingTest)
{
  // Samples in the corner of the rectangular space must be projected onto the ellipse.
  // With vx_max = 1.0 and vy_max = 1.0, a sample at (1.0, 1.0) has d^2 = 2.
  // After projection: vx = vy = 1/sqrt(2) ~= 0.7071.
  auto node = std::make_shared<nav2::LifecycleNode>("omni_elliptic_fwd_node");
  std::string name = "test";
  ParametersHandler param_handler(node, name);
  // elliptic_limits defaults to true — no explicit parameter needed

  OmniMotionModel model;
  model.initialize(&param_handler, name + ".omni");

  // vx_max = 1.0, |vx_min| = 1.0, vy_max = 1.0
  models::ControlConstraints c{1.0f, -1.0f, 1.0f, 1.9f, 3.0f, -3.0f, -3.0f, 3.0f, 3.5f};
  model.setConstraints(c, 0.05f, 0.0f, 0.0f, 0.0f, false);

  models::ControlSequence cs;
  cs.reset(3);

  // Timestep 0: exactly on the diagonal corner — must be clamped
  cs.vx(0) = 1.0f;  cs.vy(0) = 1.0f;
  // Timestep 1: inside the ellipse — must be untouched
  cs.vx(1) = 0.5f;  cs.vy(1) = 0.0f;
  // Timestep 2: negative vy corner — must be clamped symmetrically
  cs.vx(2) = 1.0f;  cs.vy(2) = -1.0f;

  model.applyConstraints(cs);

  // After projection d^2 = 1 exactly on the ellipse
  auto check_on_ellipse = [&](float vx, float vy, float vx_lim, float vy_lim) {
      float d2 = (vx / vx_lim) * (vx / vx_lim) + (vy / vy_lim) * (vy / vy_lim);
      EXPECT_NEAR(d2, 1.0f, 1e-5f);
    };

  // Timestep 0: should lie on ellipse, vx == vy (symmetric corner)
  EXPECT_NEAR(cs.vx(0), 1.0f / std::sqrt(2.0f), 1e-5f);
  EXPECT_NEAR(cs.vy(0), 1.0f / std::sqrt(2.0f), 1e-5f);
  check_on_ellipse(cs.vx(0), cs.vy(0), 1.0f, 1.0f);

  // Timestep 1: was inside the ellipse — should be unchanged
  EXPECT_NEAR(cs.vx(1), 0.5f, 1e-5f);
  EXPECT_NEAR(cs.vy(1), 0.0f, 1e-5f);

  // Timestep 2: negative vy corner, symmetric
  EXPECT_NEAR(cs.vx(2), 1.0f / std::sqrt(2.0f), 1e-5f);
  EXPECT_NEAR(cs.vy(2), -1.0f / std::sqrt(2.0f), 1e-5f);
  check_on_ellipse(cs.vx(2), cs.vy(2), 1.0f, 1.0f);
}

TEST(MotionModelTests, OmniEllipticReverseClampingTest)
{
  // Verify that the reverse-motion ellipse uses |vx_min| as the semi-axis.
  // vx_max = 1.0, vx_min = -0.5, vy_max = 0.5.
  // A reverse sample at (-0.5, 0.5) has d^2 = (0.5/0.5)^2 + (0.5/0.5)^2 = 2 → must clamp.
  auto node = std::make_shared<nav2::LifecycleNode>("omni_elliptic_rev_node");
  std::string name = "test";
  ParametersHandler param_handler(node, name);

  OmniMotionModel model;
  model.initialize(&param_handler, name + ".omni");

  // Forward: vx_max = 1.0;  Reverse: |vx_min| = 0.5;  vy_max = 0.5
  models::ControlConstraints c{1.0f, -0.5f, 0.5f, 1.9f, 3.0f, -3.0f, -3.0f, 3.0f, 3.5f};
  model.setConstraints(c, 0.05f, 0.0f, 0.0f, 0.0f, false);

  models::ControlSequence cs;
  cs.reset(2);

  // Reverse corner: both semi-axes are 0.5 so projection is symmetric
  cs.vx(0) = -0.5f;  cs.vy(0) = 0.5f;
  // Forward sample inside ellipse — untouched
  cs.vx(1) = 0.3f;   cs.vy(1) = 0.4f;

  model.applyConstraints(cs);

  // Timestep 0: projected onto ellipse, d^2 must equal 1
  float d2_rev = (cs.vx(0) / 0.5f) * (cs.vx(0) / 0.5f) +
    (cs.vy(0) / 0.5f) * (cs.vy(0) / 0.5f);
  EXPECT_NEAR(d2_rev, 1.0f, 1e-5f);
  // vx must remain negative (reverse direction preserved)
  EXPECT_LT(cs.vx(0), 0.0f);

  // Timestep 1: inside forward ellipse — unchanged
  float d2_fwd = (cs.vx(1) / 1.0f) * (cs.vx(1) / 1.0f) +
    (cs.vy(1) / 0.5f) * (cs.vy(1) / 0.5f);
  EXPECT_LT(d2_fwd, 1.0f);
  EXPECT_NEAR(cs.vx(1), 0.3f, 1e-5f);
  EXPECT_NEAR(cs.vy(1), 0.4f, 1e-5f);
}

TEST(MotionModelTests, OmniEllipticDegenerateReverseTest)
{
  // Verify that if vx_min == 0.0, the reverse ellipse has zero width.
  // Any negative vx sample must be projected strictly onto the vy axis,
  // and vy must be clamped to vy_max.
  auto node = std::make_shared<nav2::LifecycleNode>("omni_elliptic_degen_node");
  std::string name = "test";
  ParametersHandler param_handler(node, name);

  OmniMotionModel model;
  model.initialize(&param_handler, name + ".omni");

  // Reverse motion explicitly disabled (vx_min = 0.0)
  models::ControlConstraints c{1.0f, 0.0f, 0.5f, 1.9f, 3.0f, -3.0f, -3.0f, 3.0f, 3.5f};
  model.setConstraints(c, 0.05f, 0.0f, 0.0f, 0.0f, false);

  models::ControlSequence cs;
  cs.reset(2);

  // Negative vx with vy out of bounds -> should clamp vx to 0 and vy to vy_max
  cs.vx(0) = -0.5f;  cs.vy(0) = 1.0f;
  // Negative vx with vy within bounds -> should clamp vx to 0, vy unchanged
  cs.vx(1) = -0.5f;  cs.vy(1) = 0.3f;

  model.applyConstraints(cs);

  // Timestep 0: projected to vy_max
  EXPECT_NEAR(cs.vx(0), 0.0f, 1e-5f);
  EXPECT_NEAR(cs.vy(0), 0.5f, 1e-5f);

  // Timestep 1: projected to vy-axis, vy within bounds
  EXPECT_NEAR(cs.vx(1), 0.0f, 1e-5f);
  EXPECT_NEAR(cs.vy(1), 0.3f, 1e-5f);
}

TEST(MotionModelTests, OmniEllipticInsideUnchangedTest)
{
  // Any sample strictly inside the ellipse must not be modified.
  auto node = std::make_shared<nav2::LifecycleNode>("omni_elliptic_inside_node");
  std::string name = "test";
  ParametersHandler param_handler(node, name);

  OmniMotionModel model;
  model.initialize(&param_handler, name + ".omni");

  models::ControlConstraints c{0.5f, -0.35f, 0.5f, 1.9f, 3.0f, -3.0f, -3.0f, 3.0f, 3.5f};
  model.setConstraints(c, 0.05f, 0.0f, 0.0f, 0.0f, false);

  models::ControlSequence cs;
  cs.reset(4);
  // All of these are inside (vx/0.5)^2 + (vy/0.5)^2 < 1
  cs.vx(0) = 0.0f;   cs.vy(0) = 0.0f;    // origin
  cs.vx(1) = 0.3f;   cs.vy(1) = 0.0f;    // purely forward
  cs.vx(2) = 0.0f;   cs.vy(2) = 0.3f;    // purely lateral
  cs.vx(3) = 0.25f;  cs.vy(3) = 0.25f;   // diagonal but inside circle (d^2 = 0.5)

  models::ControlSequence cs_before = cs;
  model.applyConstraints(cs);

  EXPECT_NEAR(cs.vx(0), cs_before.vx(0), 1e-6f);
  EXPECT_NEAR(cs.vy(0), cs_before.vy(0), 1e-6f);
  EXPECT_NEAR(cs.vx(1), cs_before.vx(1), 1e-6f);
  EXPECT_NEAR(cs.vy(1), cs_before.vy(1), 1e-6f);
  EXPECT_NEAR(cs.vx(2), cs_before.vx(2), 1e-6f);
  EXPECT_NEAR(cs.vy(2), cs_before.vy(2), 1e-6f);
  EXPECT_NEAR(cs.vx(3), cs_before.vx(3), 1e-6f);
  EXPECT_NEAR(cs.vy(3), cs_before.vy(3), 1e-6f);
}

TEST(MotionModelTests, AckermannTest)
{
  models::ControlSequence control_sequence;
  models::State state;
  int batches = 1000;
  int timesteps = 50;
  control_sequence.reset(timesteps);  // populates with zeros
  state.reset(batches, timesteps);  // populates with zeros
  auto node = std::make_shared<nav2::LifecycleNode>("my_node");
  std::string name = "test";
  ParametersHandler param_handler(node, name);
  std::unique_ptr<AckermannMotionModel> model =
    std::make_unique<AckermannMotionModel>();
  // Initialize the plugin: parameters live under "test.ackermann"
  model->initialize(&param_handler, name + ".ackermann");

  // Check that predict properly populates the trajectory velocities with the control velocities
  state.cvx = 10 * Eigen::ArrayXXf::Ones(batches, timesteps);
  state.cvy = 5 * Eigen::ArrayXXf::Ones(batches, timesteps);
  state.cwz = 1 * Eigen::ArrayXXf::Ones(batches, timesteps);

  // Manually set state index 0 from initial conditions which would be the speed of the robot
  state.vx.col(0) = 10;
  state.wz.col(0) = 1;

  model->predict(state);

  EXPECT_TRUE(state.vx.isApprox(state.cvx));
  EXPECT_TRUE(state.vy.isApprox(Eigen::ArrayXXf::Zero(batches, timesteps)));  // non-holonomic
  EXPECT_TRUE(state.wz.isApprox(state.cwz));

  // Check that application of constraints are non-empty for Ackermann Drive
  for (unsigned int i = 0; i != control_sequence.vx.rows(); i++) {
    control_sequence.vx(i) = i * i * i;
    control_sequence.wz(i) = i * i * i * i;
  }

  models::ControlSequence initial_control_sequence = control_sequence;
  model->applyConstraints(control_sequence);
  // VX equal since this doesn't change, the WZ is reduced if breaking the constraint
  EXPECT_TRUE(initial_control_sequence.vx.isApprox(control_sequence.vx));
  EXPECT_FALSE(initial_control_sequence.wz.isApprox(control_sequence.wz));
  for (unsigned int i = 1; i != control_sequence.wz.rows(); i++) {
    EXPECT_GT(control_sequence.wz(i), 0.0);
  }

  // Now, check the specifics of the minimum curvature constraint
  EXPECT_NEAR(model->getMinTurningRadius(), 0.2, 1e-6);
  for (unsigned int i = 1; i != control_sequence.vx.rows(); i++) {
    EXPECT_TRUE(fabs(control_sequence.vx(i)) / fabs(control_sequence.wz(i)) >= 0.2);
  }

  // Check that Ackermann Drive is properly non-holonomic and parameterized
  EXPECT_EQ(model->isHolonomic(), false);

  // Check it cleanly destructs
  model.reset();
}

TEST(MotionModelTests, AckermannReversingTest)
{
  models::ControlSequence control_sequence;
  models::ControlSequence control_sequence2;
  models::State state;
  int batches = 1000;
  int timesteps = 50;
  control_sequence.reset(timesteps);  // populates with zeros
  control_sequence2.reset(timesteps);  // populates with zeros
  state.reset(batches, timesteps);  // populates with zeros
  auto node = std::make_shared<nav2::LifecycleNode>("my_node");
  std::string name = "test";
  ParametersHandler param_handler(node, name);
  std::unique_ptr<AckermannMotionModel> model =
    std::make_unique<AckermannMotionModel>();
  // Initialize the plugin: parameters live under "test.ackermann"
  model->initialize(&param_handler, name + ".ackermann");

  // Check that predict properly populates the trajectory velocities with the control velocities
  state.cvx = 10 * Eigen::ArrayXXf::Ones(batches, timesteps);
  state.cvy = 5 * Eigen::ArrayXXf::Ones(batches, timesteps);
  state.cwz = 1 * Eigen::ArrayXXf::Ones(batches, timesteps);

  // Manually set state index 0 from initial conditions which would be the speed of the robot
  state.vx.col(0) = 10;
  state.wz.col(0) = 1;

  model->predict(state);

  EXPECT_TRUE(state.vx.isApprox(state.cvx));
  EXPECT_TRUE(state.vy.isApprox(Eigen::ArrayXXf::Zero(batches, timesteps)));  // non-holonomic
  EXPECT_TRUE(state.wz.isApprox(state.cwz));

  // Check that application of constraints are non-empty for Ackermann Drive
  for (unsigned int i = 0; i != control_sequence.vx.rows(); i++) {
    float idx = static_cast<float>(i);
    control_sequence.vx(i) = -idx * idx * idx;  // now reversing
    control_sequence.wz(i) = idx * idx * idx * idx;
  }

  models::ControlSequence initial_control_sequence = control_sequence;
  model->applyConstraints(control_sequence);
  // VX equal since this doesn't change, the WZ is reduced if breaking the constraint
  EXPECT_TRUE(initial_control_sequence.vx.isApprox(control_sequence.vx));
  EXPECT_FALSE(initial_control_sequence.wz.isApprox(control_sequence.wz));
  for (unsigned int i = 1; i != control_sequence.wz.rows(); i++) {
    EXPECT_GT(control_sequence.wz(i), 0.0);
  }

  // Repeat with negative rotation direction
  for (unsigned int i = 0; i != control_sequence2.vx.rows(); i++) {
    float idx = static_cast<float>(i);
    control_sequence2.vx(i) = -idx * idx * idx;  // now reversing
    control_sequence2.wz(i) = -idx * idx * idx * idx;
  }

  models::ControlSequence initial_control_sequence2 = control_sequence2;
  model->applyConstraints(control_sequence2);
  // VX equal since this doesn't change, the WZ is reduced if breaking the constraint
  EXPECT_TRUE(initial_control_sequence2.vx.isApprox(control_sequence2.vx));
  EXPECT_FALSE(initial_control_sequence2.wz.isApprox(control_sequence2.wz));
  for (unsigned int i = 1; i != control_sequence2.wz.rows(); i++) {
    EXPECT_LT(control_sequence2.wz(i), 0.0);
  }

  // Now, check the specifics of the minimum curvature constraint
  EXPECT_NEAR(model->getMinTurningRadius(), 0.2, 1e-6);
  for (unsigned int i = 1; i != control_sequence2.vx.rows(); i++) {
    EXPECT_TRUE(fabs(control_sequence2.vx(i)) / fabs(control_sequence2.wz(i)) >= 0.2);
  }

  // Check that Ackermann Drive is properly non-holonomic and parameterized
  EXPECT_EQ(model->isHolonomic(), false);

  // Check it cleanly destructs
  model.reset();
}

static models::ControlConstraints unboundedConstraints()
{
  return {100, -100, 100, 100, 1000, -1000, -1000, 1000, 1000};
}

TEST(MotionModelTests, DelayReplayAllAxes)
{
  // Omni model exercises all three axes (vx, vy, wz) in a single test.
  // delay_vx = 0.10s -> offset 2; delay_wz = 0.15s -> offset 3.
  models::State state;
  state.reset(8, 20);

  OmniMotionModel model;

  // Set model_dt, model_delay_vx, model_delay_vy, model_delay_wz = 0.05f, 0.10f, 0.10f, 0.15f;
  model.setConstraints(unboundedConstraints(), 0.05f, 0.10f, 0.10f, 0.15f, false);

  // Ring-buffer size of vx/vy/wz = 2/2/3 steps
  // After 3 pushes per axis the rings hold the most-recent 2/2/3 values.
  model.pushCommandHistory(1.0f, 2.0f, 3.0f);
  model.pushCommandHistory(5.0f, 6.0f, 7.0f);
  model.pushCommandHistory(9.0f, 8.0f, 4.0f);

  model.predict(state);

  // predict() populates starting at column 1; column 0 holds the current
  // measurement / last command set by the optimizer before predict() runs.
  EXPECT_TRUE(state.vx.col(0).isApproxToConstant(0.0f));
  EXPECT_TRUE(state.vy.col(0).isApproxToConstant(0.0f));
  EXPECT_TRUE(state.wz.col(0).isApproxToConstant(0.0f));

  EXPECT_TRUE(state.vx.col(1).isApproxToConstant(9.0f));
  EXPECT_TRUE(state.vy.col(1).isApproxToConstant(8.0f));
  EXPECT_TRUE(state.wz.col(1).isApproxToConstant(7.0f));

  // The wz buffer has a size of 3, so check the last value as well
  EXPECT_TRUE(state.wz.col(2).isApproxToConstant(4.0f));

  // Outside of the delay window are still zero
  EXPECT_TRUE(state.vx.col(2).isApproxToConstant(0.0f));
  EXPECT_TRUE(state.vy.col(2).isApproxToConstant(0.0f));
  EXPECT_TRUE(state.vx.col(3).isApproxToConstant(0.0f));
  EXPECT_TRUE(state.vy.col(3).isApproxToConstant(0.0f));
  EXPECT_TRUE(state.wz.col(3).isApproxToConstant(0.0f));
}

TEST(MotionModelTests, DelayVyIgnoredOnDiffDrive)
{
  models::State state;
  state.reset(8, 20);

  DiffDriveMotionModel model;
  model.setConstraints(unboundedConstraints(), 0.05f, 0.0f, 0.10f, 0.0f, false);
  model.pushCommandHistory(0.0f, 99.0f, 0.0f);
  model.pushCommandHistory(0.0f, 99.0f, 0.0f);

  EXPECT_NO_THROW(model.predict(state));
  EXPECT_TRUE(state.vy.isApprox(Eigen::ArrayXXf::Zero(8, 20)));
}

TEST(MotionModelTests, DelayClearCommandHistory)
{
  models::State state;
  state.reset(8, 20);

  DiffDriveMotionModel model;
  model.setConstraints(unboundedConstraints(), 0.05f, 0.10f, 0.0f, 0.15f, false);
  model.pushCommandHistory(7.0f, 0.0f, 7.0f);
  model.clearCommandHistory();
  model.predict(state);

  EXPECT_TRUE(state.vx.col(1).isApproxToConstant(0.0f));
  EXPECT_TRUE(state.wz.col(1).isApproxToConstant(0.0f));
  EXPECT_TRUE(state.wz.col(2).isApproxToConstant(0.0f));
}

TEST(MotionModelTests, DelayZeroSkipsShift)
{
  models::State state;
  state.reset(8, 20);

  // Set non-zero velocity and check that it remains unchanged after predict()
  state.vx.col(0).setConstant(2.0f);
  state.cvx.setConstant(2.0f);

  DiffDriveMotionModel model;
  model.setConstraints(unboundedConstraints(), 0.05f, 0.0f, 0.0f, 0.0f, false);
  EXPECT_NO_THROW(model.pushCommandHistory(123.0f, 0.0f, 0.0f));
  model.predict(state);

  EXPECT_TRUE(state.vx.isApproxToConstant(2.0f));
}

TEST(MotionModelTests, ClampRawControlsRewritesRawCommand)
{
  models::State state;
  state.reset(1, 3);

  // ax_max/ax_min/az_max = 1.0, so each step can only change velocity by
  // model_dt * 1.0 = 1.0, well below the 5.0 raw commands set below
  models::ControlConstraints tight_constraints{
    100.0f, -100.0f, 100.0f, 100.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f};

  OmniMotionModel model;
  model.setConstraints(tight_constraints, 1.0f, 0.0f, 0.0f, 0.0f, true);

  state.cvx.col(0).setConstant(5.0f);
  state.cvx.col(1).setConstant(5.0f);
  state.cwz.col(0).setConstant(5.0f);
  state.cwz.col(1).setConstant(5.0f);

  model.predict(state);

  EXPECT_TRUE(state.vx.col(1).isApproxToConstant(1.0f));
  EXPECT_TRUE(state.vx.col(2).isApproxToConstant(2.0f));
  EXPECT_TRUE(state.wz.col(1).isApproxToConstant(1.0f));
  EXPECT_TRUE(state.wz.col(2).isApproxToConstant(2.0f));

  // clamp_raw_controls=true applies acceleration limits on the raw commands as well
  EXPECT_TRUE(state.cvx.col(0).isApproxToConstant(1.0f));
  EXPECT_TRUE(state.cvx.col(1).isApproxToConstant(2.0f));
  EXPECT_TRUE(state.cwz.col(0).isApproxToConstant(1.0f));
  EXPECT_TRUE(state.cwz.col(1).isApproxToConstant(2.0f));
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  rclcpp::init(0, nullptr);

  int result = RUN_ALL_TESTS();

  rclcpp::shutdown();

  return result;
}

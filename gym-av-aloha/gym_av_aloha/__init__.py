from gymnasium.envs.registration import register

register(
    id="gym_av_aloha/av-aloha-v1",
    entry_point=f"gym_av_aloha.env.sim_env:AVAlohaEnv",
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
)

register(
    id="gym_av_aloha/peg-insertion-v1",
    entry_point=f"gym_av_aloha.env.tasks.peg_insertion_env:PegInsertionEnv",
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
)

register(
    id="gym_av_aloha/cube-transfer-v1",
    entry_point=f"gym_av_aloha.env.tasks.cube_transfer_env:CubeTransferEnv",
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
)

register(
    id="gym_av_aloha/color-cubes-v1",
    entry_point=f"gym_av_aloha.env.tasks.color_cubes_env:ColorCubesEnv",
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
)

register(
    id="gym_av_aloha/thread-needle-v1",
    entry_point=f"gym_av_aloha.env.tasks.thread_needle_env:ThreadNeedleEnv",
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
)

register(
    id="gym_av_aloha/hook-package-v1",
    entry_point=f"gym_av_aloha.env.tasks.hook_package_env:HookPackageEnv",
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
)

register(
    id="gym_av_aloha/pour-test-tube-v1",
    entry_point=f"gym_av_aloha.env.tasks.pour_test_tube_env:PourTestTubeEnv",
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
)

register(
    id="gym_av_aloha/slot-insertion-v1",
    entry_point=f"gym_av_aloha.env.tasks.slot_insertion_env:SlotInsertionEnv",
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
)

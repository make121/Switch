"""Continuation restores optimizer state and starts after the saved update."""
import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf
from humanoidverse.agents.ppo.ppo_mimic import PPO


class CheckpointResumeTest(unittest.TestCase):
    def test_next_iteration_is_opt_in_and_optimizer_is_restored(self):
        source = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(source.parameters(), lr=1e-4)
        source(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model_59999.pt'
            torch.save({'model_state_dict': source.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'iter': 59999, 'infos': {'continued': True}}, path)
            for next_iteration in (False, True):
                agent = PPO.__new__(PPO)
                agent.device = 'cpu'
                agent.config = OmegaConf.create({'resume_next_iteration': next_iteration})
                agent.alg = torch.nn.Linear(2, 1)
                agent.optimizer = torch.optim.AdamW(agent.alg.parameters(), lr=.01)
                agent.load_optimizer = True
                info = agent.load(path)
                self.assertEqual(info, {'continued': True})
                self.assertEqual(agent.current_learning_iteration, 60000 if next_iteration else 59999)
                self.assertEqual(agent.learning_rate, 1e-4)
                self.assertEqual(agent.optimizer.param_groups[0]['lr'], 1e-4)
                self.assertEqual({int(v['step']) for v in agent.optimizer.state.values()}, {1})
                for key, value in agent.alg.state_dict().items():
                    torch.testing.assert_close(value, source.state_dict()[key])


if __name__ == '__main__':
    unittest.main()

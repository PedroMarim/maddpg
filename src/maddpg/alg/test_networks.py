import torch
from maddpg.alg.networks import (
    DiscreteActor,
    gumbel_softmax_sample,
    select_discrete_action,
)


def main():
    torch.manual_seed(0)

    # Example: adversary obs_dim=8, goods obs_dim=10
    actor_adv = DiscreteActor(obs_dim=8, num_actions=5)
    actor_good = DiscreteActor(obs_dim=10, num_actions=5)

    obs_adv = torch.randn(8)
    obs_good = torch.randn(10)

    # Forward pass -> logits
    logits_adv = actor_adv(obs_adv.unsqueeze(0))
    logits_good = actor_good(obs_good.unsqueeze(0))
    print("logits shapes:", logits_adv.shape, logits_good.shape)

    # Gumbel-softmax (training-time action vector)
    a_soft = gumbel_softmax_sample(logits_adv, tau=1.0, hard=False)
    print("soft action shape:", a_soft.shape, "sum:", a_soft.sum(dim=-1))

    # Integer action for env
    a_int = select_discrete_action(actor_adv, obs_adv, explore=True)
    print("env action int:", a_int)

    print("networks test ok")


if __name__ == "__main__":
    main()

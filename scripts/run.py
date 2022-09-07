import imp
import PIL
import os
import argparse, os, sys, glob, random
import torch
import numpy as np
import copy
import yaml
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.modules.encoders.modules import FrozenCLIPEmbedder
from functools import partial
from ldm.modules.embedding_manager import EmbeddingManager

def create_embedder(embeddings):	
	print("setting up embeddings")

	def get_clip_token_for_string(tokenizer, string):
		print('checking existing tokens')
		print(tokenizer.tokenize(string))
		batch_encoding = tokenizer(string, truncation=True, max_length=77, return_length=True,
			return_overflowing_tokens=False, padding="max_length",
			return_tensors="pt")	    
		tokens = batch_encoding["input_ids"]

		print(f'returning {tokens[0, 1]}')
		if torch.count_nonzero(tokens - 49407) == 2:			
			return tokens[0, 1]
		
		raise Exception('invalid encoding')

	embedder = FrozenCLIPEmbedder().cuda()
	EmbeddingManagerPartial = partial(EmbeddingManager, embedder, [])
	string_to_param_dict = torch.nn.ParameterDict()
	string_to_token_dict = {}

	def create_embedding(path, placeholder):
		manager = EmbeddingManagerPartial()
		manager.load(path)		
		string_to_token_dict[placeholder] = get_clip_token_for_string(embedder.tokenizer, placeholder)					
		string_to_param_dict[placeholder] = manager.string_to_param_dict['*']
	
	for embedding in embeddings:
		create_embedding(embedding['file'], embedding['token'])
	
	embedder = EmbeddingManagerPartial()
	embedder.string_to_param_dict = string_to_param_dict
	embedder.string_to_token_dict = string_to_token_dict
	return embedder

def load_model_from_config(config, ckpt, embeddings, verbose=False):
	embedder = create_embedder(embeddings)
	print(f"Loading model from {ckpt}")
	pl_sd = torch.load(ckpt, map_location="cpu")
	if "global_step" in pl_sd:
		print(f"Global Step: {pl_sd['global_step']}")
	sd = pl_sd["state_dict"]
	model = instantiate_from_config(config.model)
	m, u = model.load_state_dict(sd, strict=False)
	if len(m) > 0 and verbose:
		print("missing keys:")
		print(m)
	if len(u) > 0 and verbose:
		print("unexpected keys:")
		print(u)	

	model.cuda()
	model.eval()

	model.embedding_manager = embedder

	return model

def load_img(path):
	image = Image.open(path).convert("RGB")
	w, h = image.size
	print(f"loaded input image of size ({w}, {h}) from {path}")
	w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
	image = image.resize((w, h), resample=PIL.Image.LANCZOS)
	image = np.array(image).astype(np.float32) / 255.0
	image = image[None].transpose(0, 3, 1, 2)
	image = torch.from_numpy(image)
	return 2.*image - 1.

def convert_img(pil_img):
	image = pil_img.convert("RGB")
	w, h = image.size
	print(f"loaded input image of size ({w}, {h})")
	w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
	image = image.resize((w, h), resample=PIL.Image.LANCZOS)
	image = np.array(image).astype(np.float32) / 255.0
	image = image[None].transpose(0, 3, 1, 2)
	image = torch.from_numpy(image)
	return 2.*image - 1.


def load_model(config, ckpt, embeddings, half = False):
	config = OmegaConf.load(f"{config}")
	model = load_model_from_config(config, f"{ckpt}", embeddings)
	
	if half:
		return model.half()#.to(device)
	else:
		return model

def run_img(model, input, prompt, steps, strength, scale, half = False):	
	input = convert_img(input)
	
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	model.to(device)

	ddim_eta = 0
	sampler = DDIMSampler(model)	
	batch_size = 1	
	data = [batch_size * [prompt]]
			
	if half:
		input = input.half()
	init_image = input.to(device)
	init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
	# move to latent space
	init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))

	sampler.make_schedule(ddim_num_steps=steps, ddim_eta=ddim_eta, verbose=False)

	assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
	t_enc = int(strength * steps)
	print(f"target t_enc is {t_enc} steps")
	#seed = random.randint(0, 4294960000)
	seed = 42
	seed_everything(seed)
	
	precision_scope = autocast
	with torch.no_grad():
		with precision_scope("cuda"):
			with model.ema_scope():
				all_samples = list()				
				for prompts in tqdm(data, desc="data"):					
					uc = None
					if scale != 1.0:
						uc = model.get_learned_conditioning(batch_size * [""])
					if isinstance(prompts, tuple):
						prompts = list(prompts)
					c = model.get_learned_conditioning(prompts)

					# encode (scaled latent)
					z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
					# decode it
					samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
											 unconditional_conditioning=uc,)
					x_samples = model.decode_first_stage(samples)
					x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)					

					for x_sample in x_samples:
						x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')						
						return Image.fromarray(x_sample.astype(np.uint8))

	return None


def run(model, input_mode, opt, after_run):
	tic = time.time()	

	session = opt.session_name	
	print("Starting Session #" + str(session))

	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	model.to(device)	

	if opt.plms:
		sampler = PLMSSampler(model)
	else:
		sampler = DDIMSampler(model)

	if not opt.fixed_out:
		os.makedirs(opt.outdir, exist_ok=True)
		outpath = opt.outdir

	batch_size = opt.n_samples
	n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
	if not opt.from_file:
		prompt = opt.prompt
		assert prompt is not None
		data = [batch_size * [prompt]]

	else:
		print(f"reading prompts from {opt.from_file}")
		with open(opt.from_file, "r") as f:
			data = f.read().splitlines()
			data = list(chunk(data, batch_size))

	if not opt.fixed_out:
		sample_path = os.path.join(outpath, "sessions")
		os.makedirs(sample_path, exist_ok=True)
		session_parent = os.path.join(sample_path, f"{session}")
		session_path = os.path.join(session_parent, "0")
		n = 1
		while os.path.exists(session_path):
			session_path = os.path.join(session_parent, f"{n}")
			n += 1

		os.makedirs(session_path, exist_ok=True)
		config_path = os.path.join(session_path, "config.yaml")
		with open(config_path, 'w') as file:
			yaml.dump(opt, file)

		base_count = len(os.listdir(session_path))
		grid_count = len(os.listdir(session_path)) - 1

	if input_mode == "img2img":
		assert os.path.isfile(opt.init_img)
		init_image = load_img(opt.init_img).to(device)
		init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
		init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

		sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

		assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
		t_enc = int(opt.strength * opt.ddim_steps)
		print(f"target t_enc is {t_enc} steps")
	else:
		start_code = None
		if opt.fixed_code:
			start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

	precision_scope = autocast if opt.precision=="autocast" else nullcontext
	with torch.no_grad():
		with precision_scope("cuda"):
			with model.ema_scope():
				all_samples = list()
				for n in trange(opt.n_iter, desc="Sampling"):
					for prompts in tqdm(data, desc="data"):
						seed_everything(opt.seed)
						uc = None
						if opt.scale != 1.0:
							uc = model.get_learned_conditioning(batch_size * [""])
						if isinstance(prompts, tuple):
							prompts = list(prompts)
						c = model.get_learned_conditioning(prompts)

						if input_mode == "img2img":
							# encode (scaled latent)
							z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
							# decode it
							samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
													 unconditional_conditioning=uc,)
							x_samples = model.decode_first_stage(samples)
							x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
						else:
							shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
							samples, _ = sampler.sample(S=opt.ddim_steps,
															 conditioning=c,
															 batch_size=opt.n_samples,
															 shape=shape,
															 verbose=False,
															 unconditional_guidance_scale=opt.scale,
															 unconditional_conditioning=uc,
															 eta=opt.ddim_eta,
															 x_T=start_code)
							x_samples = model.decode_first_stage(samples)
							x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

						if not opt.skip_save:
							for x_sample in x_samples:
								x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
								if not opt.fixed_out:
									path = os.path.join(session_path, f"{base_count:05}_seed-{opt.seed}.png")
								else:									
									out_dir = "D:/media/New folder/fs2/gen_faces"
									base_count = len(os.listdir(out_dir))
									path = os.path.join(out_dir, f"{base_count:05}.png")
									base_count += 1
								Image.fromarray(x_sample.astype(np.uint8)).save(path)								

						if not opt.skip_grid:
							all_samples.append(x_samples)
						opt.seed += 1

				if not opt.skip_grid:
					# additionally, save as grid
					grid = torch.stack(all_samples, 0)
					grid = rearrange(grid, 'n b c h w -> (n b) c h w')
					grid = make_grid(grid, nrow=n_rows)

					# to image
					grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
					Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
					grid_count += 1				


	toc = time.time()

	time_taken = (toc-tic)/60.0

	if not opt.fixed_out:
		print(("Your samples are ready in {0:.2f} minutes and waiting for you here \n" + sample_path).format(time_taken))
	after_run()

import diffusers
import torch

from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler, CogVideoXPipeline, CogVideoXTransformer3DModel

class CustomCogVideoXTransformer3DModel(CogVideoXTransformer3DModel):
    # Do not override the from_pretrained method
    # Any customization should be done after loading the model

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        clip_prompt_embeds,
        ref_img_states=None,
        timestep=None,
        attention_mask=None,
        cross_attention_kwargs=None,
        encoder_attention_mask=None,
        return_dict=False,
        customization=False,
        concatenated_all=False,
        reduce_token=False,
        add_token=False,
        zero_conv_add=False,
        vae_add=False,
        pos_embed=False,
        cross_attend=False,
    ):
        # # Use custom components if they are available
        # if hasattr(self, 'T5ProjectionLayer'):
        #     encoder_hidden_states = self.T5ProjectionLayer(encoder_hidden_states)

        # if ref_img_states is not None and hasattr(self, 'reference_vision_encoder'):
        #     vision_outputs = self.reference_vision_encoder(pixel_values=ref_img_states)
        #     image_embeds = vision_outputs.last_hidden_state
        #     if hasattr(self, 'CLIPVisionProjectionLayer'):
        #         image_embeds = self.CLIPVisionProjectionLayer(image_embeds)
        #     encoder_hidden_states = torch.cat([encoder_hidden_states, image_embeds], dim=1)
        
        return super().forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            clip_prompt_embeds=clip_prompt_embeds,
            ref_img_states=ref_img_states,
            timestep=timestep,
            # attention_mask=attention_mask,
            # cross_attention_kwargs=cross_attention_kwargs,
            # encoder_attention_mask=encoder_attention_mask,
            return_dict=return_dict,
            customization=customization,
            eval=True,
            concatenated_all=concatenated_all,
            reduce_token=reduce_token,
            add_token=add_token,
            zero_conv_add=zero_conv_add,
            vae_add=vae_add,
            pos_embed=pos_embed,
            cross_attend=cross_attend,
            # eval=True,
            **kwargs,
        )

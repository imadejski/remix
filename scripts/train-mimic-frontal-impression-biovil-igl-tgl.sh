LOSS_COMBO=igl_tgl
SECTION=impression
NAME=mimic-frontal-biovil
VERSION=$SECTION-$LOSS_COMBO

python train.py fit \
--config configs/$NAME.yaml \
--model.checkpoint_path /opt/gpudata/remix/$NAME-$VERSION \
--model.loss_combo $LOSS_COMBO \
--data.section $SECTION \
--trainer.logger.name $NAME \
--trainer.logger.version $VERSION

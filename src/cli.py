import click



@click.group()
@click.option('-e', '--evaluation', default=True, help='Choose to do evaluation or not')
@click.option('-dd', '--download_dicom', default=True, help='Download the DICOM files -will take a while!-')
@click.option('-id', '--ingest_dicom', default=True, help='Choose to ingest the DICOM metadata')
@click.option('-ix', '--ingest_xcelera', default=True, help='Choose to ingest the XCelera data')
@click.option('-i', '--ingestion', default=True, help='Ingest the DICOM metadata and XCelera data -without download-')
@click.option('-c', '--classification', default=True, help='Execute classification task')
@click.option('-s', '--segmentation', default=True, help='Execute segmentation task')
@click.option('-m', '--measurement', default=True, help='Execute measurement task')
@click.option('-fwi', '--run_pipeline_without_ingestion', default=True,
              help='Execute the complete pipeline: classification, segmentation, measurements')
@click.option('-f', '--run_pipeline', default=True,
             help='Execute the complete pipeline: ingestion, classification, segmentation, measurements')
def cli():
    pass

@cli.command()
def download_dicom(download_dicom):
    click.echo("### downlaod dicom")



